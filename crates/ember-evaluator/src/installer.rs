use crate::{Error, Hello, Result, PROTOCOL_VERSION};
use ed25519_dalek::{Signature, Verifier, VerifyingKey};
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    fs,
    io::{Read, Write},
    path::{Component, Path, PathBuf},
    time::Duration,
};

const MAX_ARCHIVE: u64 = 2 * 1024 * 1024 * 1024;
const MAX_EXPANDED: u64 = 6 * 1024 * 1024 * 1024;
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignedManifest {
    pub payload: String,
    pub signature: String,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Manifest {
    pub version: String,
    pub target: String,
    pub protocol_version: u32,
    pub catalog_version: String,
    pub url: String,
    pub sha256: String,
    pub size: u64,
    pub entrypoint: String,
    #[serde(default)]
    pub args: Vec<String>,
}
#[derive(Clone)]
pub struct ModuleManager {
    root: PathBuf,
    catalog: String,
    key: Option<[u8; 32]>,
}
pub fn target() -> String {
    format!("{}-{}", std::env::consts::ARCH, std::env::consts::OS)
}
fn relative(path: &str) -> Result<PathBuf> {
    if path.is_empty()
        || path.contains('\\')
        || path.contains(':')
        || Path::new(path)
            .components()
            .any(|c| !matches!(c, Component::Normal(_)))
    {
        return Err(Error::Invalid(
            "Module path must be a safe relative path".into(),
        ));
    }
    Ok(PathBuf::from(path))
}
impl ModuleManager {
    pub fn new(root: PathBuf, catalog: String, public_key: Option<&str>) -> Result<Self> {
        fs::create_dir_all(&root)?;
        let key = public_key
            .map(|key| {
                let bytes = hex::decode(key).map_err(|e| Error::Invalid(e.to_string()))?;
                bytes
                    .try_into()
                    .map_err(|_| Error::Invalid("Ed25519 public key must be 32 bytes".into()))
            })
            .transpose()?;
        Ok(Self { root, catalog, key })
    }
    fn verify(&self, signed: &SignedManifest) -> Result<Manifest> {
        let key = self
            .key
            .ok_or_else(|| Error::Invalid("No evaluator release public key configured".into()))?;
        let signature = Signature::from_slice(
            &hex::decode(&signed.signature).map_err(|e| Error::Invalid(e.to_string()))?,
        )
        .map_err(|e| Error::Invalid(e.to_string()))?;
        VerifyingKey::from_bytes(&key)
            .map_err(|e| Error::Invalid(e.to_string()))?
            .verify(signed.payload.as_bytes(), &signature)
            .map_err(|_| Error::Invalid("Invalid module signature".into()))?;
        let manifest: Manifest = serde_json::from_str(&signed.payload)?;
        let version =
            semver::Version::parse(&manifest.version).map_err(|e| Error::Invalid(e.to_string()))?;
        if version.to_string() != manifest.version || manifest.version.contains(['/', '\\']) {
            return Err(Error::Invalid("Invalid module version".into()));
        }
        Hello {
            protocol_version: manifest.protocol_version,
            catalog_version: manifest.catalog_version.clone(),
            engine_version: manifest.version.clone(),
        }
        .validate(&self.catalog)?;
        if manifest.target != target()
            || manifest.size == 0
            || manifest.size > MAX_ARCHIVE
            || manifest.sha256.len() != 64
            || hex::decode(&manifest.sha256).is_err()
        {
            return Err(Error::Invalid("Invalid module target, size or hash".into()));
        }
        relative(&manifest.entrypoint)?;
        if manifest.args.len() > 16 || manifest.args.iter().any(|s| s.len() > 4096) {
            return Err(Error::Invalid("Invalid module arguments".into()));
        }
        Ok(manifest)
    }
    pub fn active(&self) -> Result<Option<(Manifest, crate::local::Launch)>> {
        let path = self.root.join("active.json");
        if !path.exists() {
            return Ok(None);
        }
        let signed: SignedManifest = serde_json::from_slice(&fs::read(path)?)?;
        let manifest = self.verify(&signed)?;
        let dir = self.root.join(&manifest.version);
        let launch = crate::local::Launch {
            executable: dir.join(relative(&manifest.entrypoint)?),
            args: manifest.args.clone(),
            working_dir: dir,
        };
        if !launch.executable.is_file() {
            return Err(Error::Invalid(
                "Active evaluator entrypoint is missing".into(),
            ));
        }
        Ok(Some((manifest, launch)))
    }
    pub fn launch(&self, version: &str) -> Result<crate::local::Launch> {
        relative(version)?;
        let dir = self.root.join(version);
        let signed: SignedManifest = serde_json::from_slice(&fs::read(dir.join("module.json"))?)?;
        let manifest = self.verify(&signed)?;
        if manifest.version != version {
            return Err(Error::Invalid("Module version mismatch".into()));
        }
        Ok(crate::local::Launch {
            executable: dir.join(relative(&manifest.entrypoint)?),
            args: manifest.args,
            working_dir: dir,
        })
    }
    pub fn installed(&self) -> Result<Vec<String>> {
        let mut versions = vec![];
        for entry in fs::read_dir(&self.root)? {
            let entry = entry?;
            if entry.file_type()?.is_dir() && entry.path().join("module.json").is_file() {
                versions.push(entry.file_name().to_string_lossy().into_owned());
            }
        }
        versions.sort();
        Ok(versions)
    }
    pub fn activate(&self, version: &str) -> Result<()> {
        relative(version)?;
        let signed: SignedManifest =
            serde_json::from_slice(&fs::read(self.root.join(version).join("module.json"))?)?;
        let manifest = self.verify(&signed)?;
        if manifest.version != version
            || !self
                .root
                .join(version)
                .join(relative(&manifest.entrypoint)?)
                .is_file()
        {
            return Err(Error::Invalid("Incomplete installed evaluator".into()));
        }
        self.write_active(&signed)
    }
    fn write_active(&self, signed: &SignedManifest) -> Result<()> {
        let mut temp = tempfile::NamedTempFile::new_in(&self.root)?;
        temp.write_all(&serde_json::to_vec(signed)?)?;
        temp.as_file().sync_all()?;
        temp.persist(self.root.join("active.json"))
            .map_err(|e| e.error)?;
        #[cfg(unix)]
        fs::File::open(&self.root)?.sync_all()?;
        Ok(())
    }

    pub async fn download(&self, manifest_url: &str) -> Result<String> {
        let url = reqwest::Url::parse(manifest_url).map_err(|e| Error::Invalid(e.to_string()))?;
        check_url(&url)?;
        let client = reqwest::Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .timeout(Duration::from_secs(300))
            .build()?;
        let response = client.get(url).send().await?.error_for_status()?;
        let mut data = Vec::new();
        let mut stream = response.bytes_stream();
        while let Some(chunk) = stream.next().await {
            let chunk = chunk?;
            if data.len() + chunk.len() > 64 * 1024 {
                return Err(Error::Invalid("Manifest too large".into()));
            }
            data.extend_from_slice(&chunk);
        }
        let signed: SignedManifest = serde_json::from_slice(&data)?;
        let manifest = self.verify(&signed)?;
        let url = reqwest::Url::parse(&manifest.url).map_err(|e| Error::Invalid(e.to_string()))?;
        check_url(&url)?;
        let archive = self
            .root
            .join(format!("download-{}.zip", uuid::Uuid::new_v4()));
        let outcome = async {
            let mut file = fs::File::create(&archive)?;
            let mut stream = client
                .get(url)
                .send()
                .await?
                .error_for_status()?
                .bytes_stream();
            let mut count = 0u64;
            while let Some(chunk) = stream.next().await {
                let chunk = chunk?;
                count += chunk.len() as u64;
                if count > manifest.size {
                    return Err(Error::Invalid(
                        "Module download exceeded signed size".into(),
                    ));
                }
                file.write_all(&chunk)?;
            }
            file.sync_all()?;
            let manager = self.clone();
            let source = archive.clone();
            tokio::task::spawn_blocking(move || manager.install(&signed, &source))
                .await
                .map_err(|e| Error::Invalid(e.to_string()))?
        }
        .await;
        let _ = fs::remove_file(archive);
        outcome
    }
    pub fn import(&self, manifest_path: &Path, archive: &Path) -> Result<String> {
        if fs::metadata(manifest_path)?.len() > 64 * 1024 {
            return Err(Error::Invalid("Manifest too large".into()));
        }
        let signed = serde_json::from_slice(&fs::read(manifest_path)?)?;
        self.install(&signed, archive)
    }
    fn install(&self, signed: &SignedManifest, archive: &Path) -> Result<String> {
        let manifest = self.verify(signed)?;
        if fs::metadata(archive)?.len() != manifest.size {
            return Err(Error::Invalid("Module size mismatch".into()));
        }
        let mut hash = Sha256::new();
        let mut file = fs::File::open(archive)?;
        let mut buffer = [0; 65536];
        loop {
            let size = file.read(&mut buffer)?;
            if size == 0 {
                break;
            }
            hash.update(&buffer[..size]);
        }
        if hex::encode(hash.finalize()) != manifest.sha256 {
            return Err(Error::Invalid("Module checksum mismatch".into()));
        }
        let destination = self.root.join(&manifest.version);
        if destination.exists() {
            return Err(Error::Invalid(
                "Module version already installed; activate it or publish a new version".into(),
            ));
        }
        let stage = self.root.join(format!(".stage-{}", uuid::Uuid::new_v4()));
        fs::create_dir(&stage)?;
        let outcome = (|| {
            let mut zip = zip::ZipArchive::new(fs::File::open(archive)?)
                .map_err(|e| Error::Invalid(e.to_string()))?;
            if zip.len() > 100_000 {
                return Err(Error::Invalid("Too many module files".into()));
            }
            let mut expanded = 0u64;
            for index in 0..zip.len() {
                let mut entry = zip
                    .by_index(index)
                    .map_err(|e| Error::Invalid(e.to_string()))?;
                let mode = entry.unix_mode().unwrap_or(0);
                if mode & 0o170000 == 0o120000 {
                    return Err(Error::Invalid("Module symlinks are forbidden".into()));
                }
                let path = stage.join(relative(entry.name().trim_end_matches('/'))?);
                expanded = expanded
                    .checked_add(entry.size())
                    .ok_or_else(|| Error::Invalid("Module size overflow".into()))?;
                if expanded > MAX_EXPANDED {
                    return Err(Error::Invalid("Expanded module too large".into()));
                }
                if entry.is_dir() {
                    fs::create_dir_all(&path)?;
                    continue;
                }
                if let Some(parent) = path.parent() {
                    fs::create_dir_all(parent)?;
                }
                let mut output = fs::OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .open(&path)?;
                let expected = entry.size();
                let copied = std::io::copy(&mut (&mut entry).take(expected + 1), &mut output)?;
                if copied != expected {
                    return Err(Error::Invalid("Module entry size mismatch".into()));
                }
                #[cfg(unix)]
                {
                    use std::os::unix::fs::PermissionsExt;
                    fs::set_permissions(
                        &path,
                        fs::Permissions::from_mode(if mode & 0o111 != 0 { 0o755 } else { 0o644 }),
                    )?;
                }
            }
            if !stage.join(relative(&manifest.entrypoint)?).is_file() {
                return Err(Error::Invalid("Module entrypoint missing".into()));
            }
            fs::write(stage.join("module.json"), serde_json::to_vec(signed)?)?;
            fs::rename(&stage, destination)?;
            Ok(manifest.version.clone())
        })();
        if stage.exists() {
            let _ = fs::remove_dir_all(stage);
        }
        outcome
    }
}
fn check_url(url: &reqwest::Url) -> Result<()> {
    if url.scheme() != "https"
        && !(url.scheme() == "http"
            && matches!(url.host_str(), Some("127.0.0.1" | "localhost" | "[::1]")))
    {
        return Err(Error::Invalid("Module download requires HTTPS".into()));
    }
    Ok(())
}
pub fn protocol_version() -> u32 {
    PROTOCOL_VERSION
}

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::{Signer, SigningKey};
    fn fixture(
        version: &str,
        name: &str,
    ) -> (tempfile::TempDir, ModuleManager, SignedManifest, PathBuf) {
        let dir = tempfile::tempdir().unwrap();
        let signing = SigningKey::from_bytes(&[7; 32]);
        let manager = ModuleManager::new(
            dir.path().join("modules"),
            "catalog".into(),
            Some(&hex::encode(signing.verifying_key().as_bytes())),
        )
        .unwrap();
        let archive = dir.path().join("module.zip");
        let mut zip = zip::ZipWriter::new(fs::File::create(&archive).unwrap());
        zip.start_file(
            name,
            zip::write::SimpleFileOptions::default().unix_permissions(0o755),
        )
        .unwrap();
        zip.write_all(b"test module").unwrap();
        zip.finish().unwrap();
        let data = fs::read(&archive).unwrap();
        let manifest = Manifest {
            version: version.into(),
            target: target(),
            protocol_version: 1,
            catalog_version: "catalog".into(),
            url: "https://example.com/module.zip".into(),
            sha256: hex::encode(Sha256::digest(&data)),
            size: data.len() as u64,
            entrypoint: "runner".into(),
            args: vec![],
        };
        let payload = serde_json::to_string(&manifest).unwrap();
        let signature = hex::encode(signing.sign(payload.as_bytes()).to_bytes());
        (dir, manager, SignedManifest { payload, signature }, archive)
    }
    #[test]
    fn signed_install_activation_and_integrity() {
        let (_dir, manager, signed, archive) = fixture("0.1.0", "runner");
        manager.install(&signed, &archive).unwrap();
        manager.activate("0.1.0").unwrap();
        assert_eq!(manager.active().unwrap().unwrap().0.version, "0.1.0");
        assert!(manager.install(&signed, &archive).is_err());
        let mut forged = signed.clone();
        forged.payload = forged.payload.replace("0.1.0", "0.2.0");
        assert!(manager.verify(&forged).is_err());
        fs::write(&archive, b"tampered").unwrap();
        assert!(manager.install(&signed, &archive).is_err());
        manager.activate("0.1.0").unwrap();
        assert!(manager.activate("../outside").is_err());
    }
    #[test]
    fn traversal_cannot_escape_staging() {
        let (dir, manager, signed, archive) = fixture("0.1.0", "../escape");
        assert!(manager.install(&signed, &archive).is_err());
        assert!(!dir.path().join("escape").exists());
        assert!(manager.active().unwrap().is_none());
    }
    #[test]
    fn protocol_and_catalog_are_independent_of_engine_version() {
        let (_dir, manager, signed, _) = fixture("9.2.1", "runner");
        assert_eq!(manager.verify(&signed).unwrap().version, "9.2.1");
        assert!(Hello {
            protocol_version: 2,
            catalog_version: "catalog".into(),
            engine_version: "9.2.1".into()
        }
        .validate("catalog")
        .is_err());
        assert!(Hello {
            protocol_version: 1,
            catalog_version: "other".into(),
            engine_version: "9.2.1".into()
        }
        .validate("catalog")
        .is_err());
    }
}
