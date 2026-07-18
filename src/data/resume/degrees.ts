export interface Degree {
  school: string;
  degree: string;
  link: string;
  year: number;
  details?: string[];
}

const degrees: Degree[] = [
  {
    school: 'University of Illinois Urbana-Champaign',
    degree: 'B.S. Electrical Engineering',
    link: 'https://illinois.edu',
    year: 2028,
    details: ['GPA: 4.00/4.00', "Dean's List 2024–2025, 2025–2026"],
  },
  {
    school: 'Zhejiang University',
    degree: 'B.Eng. Electrical Engineering',
    link: 'https://www.zju.edu.cn/english/',
    year: 2028,
    details: [
      'GPA: 3.93/4.00 — Rank 1/71',
      'National Scholarship; ZJU First-Class Scholarship; ZJU First-Class Institute Scholarship',
    ],
  },
];

export default degrees;
