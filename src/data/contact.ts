import { IconDefinition } from '@fortawesome/fontawesome-svg-core';
import { faGithub } from '@fortawesome/free-brands-svg-icons/faGithub';
import { faEnvelope } from '@fortawesome/free-regular-svg-icons/faEnvelope';

export interface ContactItem {
  link: string;
  label: string;
  icon: IconDefinition;
}

const data: ContactItem[] = [
  {
    link: 'https://github.com/hasson827',
    label: 'GitHub',
    icon: faGithub,
  },
  {
    link: 'mailto:hasson827624@gmail.com',
    label: 'Gmail',
    icon: faEnvelope,
  },
  {
    link: 'mailto:hz108@illinois.edu',
    label: 'UIUC Email',
    icon: faEnvelope,
  },
  {
    link: 'mailto:hongshuo.24@intl.zju.edu.cn',
    label: 'Zhejiang Email',
    icon: faEnvelope,
  },
];

export default data;
