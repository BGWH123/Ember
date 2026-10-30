"""Run restored task tests through the real, bounded evaluator process."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from ember_evaluator.domain import ResourceBudget
from ember_evaluator.judge import execute
from ember_evaluator import judge
from torch_judge.tasks import TASKS

# Plausible errors must fail assertions, rather than merely fail to import/run.
MUTATIONS = {
    'adamw': ('param.mul_(1.0 - self.lr * self.weight_decay)', 'param.mul_(1.0)'),
    'attention_mask': ('float("-inf")', '0.0'),
    'binary_cross_entropy': ('.mean()', '.sum()'),
    'block_topk_attention': ('block_scores.topk(top_k', 'block_scores.topk(1'),
    'bm25_rrf_retrieval': ('term_freq * (k1 + 1.0)', 'term_freq * k1'),
    'chinchilla_scaling': ('beta / (alpha + beta)', 'alpha / (alpha + beta)'),
    'cosine_similarity': ('torch.dot(a_normalized, b_normalized)', 'torch.dot(a, b)'),
    'deepseek_mtp_module': ('targets.reshape(-1))', 'targets.reshape(-1).roll(1))'),
    'generalized_advantage_estimation': ('gamma * lam * mask * gae', 'gamma * mask * gae'),
    'gru_cell': ('reset_gate * h_prev', 'h_prev'),
    'groupwise_int4_quant': (' / 15.0', ' / 16.0'),
    'jepa_and_vicreg_loss': ('(B - 1)', 'B'),
    'kabsch_pointcloud_align': ('translation = dst_center - rotation @ src_center', 'translation = dst_center - src_center'),
    'kl_divergence': ('.sum()', '.mean()'),
    'kivi_kv_cache_quant': ('q_max = 2 ** n_bits - 1', 'q_max = 2 ** n_bits'),
    'l1_regularization': ('torch.abs(w)', 'w'),
    'l2_regularization': ('0.5 * lambda_reg', 'lambda_reg'),
    'layerscale': ('out = x + scaled_branch', 'out = scaled_branch'),
    'leaky_relu': ('alpha * x', '0.0 * x'),
    'lstm_cell': ('forget_gate * c_prev + input_gate * candidate', 'input_gate * candidate'),
    'mse_loss': ('.mean()', '.sum()'),
    'mla_rope_absorption': ('math.sqrt(d_h + d_r)', 'math.sqrt(d_h)'),
    'noisy_topk_gating': ('if training:', 'if False:'),
    'one_hot_encoding': ('indices.unsqueeze(1), 1.0)', 'indices.unsqueeze(1), 0.0)'),
    'residual_connection': ('return x + f_x', 'return f_x'),
    'sequence_pack': ('mask[i, :length] = 1.0', 'mask[i, :] = 1.0'),
    'sgd_momentum': ('decayed_velocity + grad', 'grad'),
    'sigmoid': ('result[~positive] = exp_x / (1.0 + exp_x)', 'result[~positive] = 0.0'),
    'softmax_temperature': ('logits / temperature', 'logits * temperature'),
    'stochastic_depth': ('(x.shape[0],) + (1,) * (x.ndim - 1)', 'x.shape'),
    'tanh': ('out = torch.tanh(x)', 'out = x'),
    'wordpiece_tokenizer': ('range(len(word), start, -1)', 'range(start + 1, len(word) + 1)'),
}


@pytest.mark.parametrize('task_id', MUTATIONS)
def test_restored_reference_passes_real_evaluator(task_id):
    result = execute(task_id, TASKS[task_id]['solution'], None, ResourceBudget())
    assert result.outcome == 'passed', (result.error, result.details)
    assert result.passed_count == result.total_count >= 2


@pytest.mark.parametrize('task_id', MUTATIONS)
def test_restored_tests_reject_plausible_wrong_answer(task_id):
    source = TASKS[task_id]['solution']
    before, after = MUTATIONS[task_id]
    assert before in source, f'Mutation no longer matches {task_id}'
    result = execute(task_id, source.replace(before, after), None, ResourceBudget())
    assert result.outcome == 'failed', (result.error, result.details)
    assert result.error_code is None, 'A test assertion must reject the wrong answer'
    assert result.total_count > result.passed_count


def test_empty_test_placeholders_are_rejected_before_execution(monkeypatch):
    monkeypatch.setattr(judge, '_COMPILED_TESTS', {})
    monkeypatch.setattr(judge, 'TASKS', {'empty_fixture': {'tests': [{'name': 'placeholder', 'code': '  \n'}]}})
    with pytest.raises(ValueError, match='empty test placeholders'):
        judge.prewarm()
    assert judge._COMPILED_TESTS == {}
