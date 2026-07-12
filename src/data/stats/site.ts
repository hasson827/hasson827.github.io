import dayjs from 'dayjs';

import { StatData } from '../../components/Stats/types';

/* Keys match keys returned by the github api. To see everything returned by
 * the github api, run:
 * curl https://api.github.com/repos/hasson827/hasson827.github.io
 */
const data: StatData[] = [
  {
    label: 'Stars this repository has on github',
    key: 'stargazers_count',
    link: 'https://github.com/hasson827/hasson827.github.io/stargazers',
  },
  {
    label: 'Number of people watching this repository',
    key: 'subscribers_count',
    link: 'https://github.com/hasson827/hasson827.github.io/stargazers',
  },
  {
    label: 'Number of forks',
    key: 'forks',
    link: 'https://github.com/hasson827/hasson827.github.io/network',
  },
  {
    label: 'Open github issues',
    key: 'open_issues_count',
    link: 'https://github.com/hasson827/hasson827.github.io/issues',
  },
  {
    label: 'Last updated at',
    key: 'pushed_at',
    link: 'https://github.com/hasson827/hasson827.github.io/commits',
    format: (x: unknown) => dayjs(x as string).format('MMMM DD, YYYY'),
  },
];

export default data;
