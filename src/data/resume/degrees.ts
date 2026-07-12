export interface Degree {
  school: string;
  degree: string;
  link: string;
  year: number;
}

const degrees: Degree[] = [
  {
    school: 'University of Illinois Urbana-Champaign',
    degree: 'B.S. Electrical Engineering',
    link: 'https://illinois.edu',
    year: 2028,
  },
  {
    school: 'Zhejiang University',
    degree: 'B.Eng. Electrical Engineering',
    link: 'https://www.zju.edu.cn/english/',
    year: 2028,
  },
];

export default degrees;
