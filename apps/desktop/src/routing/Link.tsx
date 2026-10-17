import { Link as RouterLink, type LinkProps } from 'react-router-dom';
export default function Link({href,...props}:Omit<LinkProps,'to'> & {href:string}) {return <RouterLink to={href} {...props}/>;}
