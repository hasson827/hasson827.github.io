/**
 * Conforms to https://jsonresume.org/schema/
 */
export interface Position {
  name: string;
  position: string;
  url: string;
  startDate: string;
  endDate?: string;
  summary?: string;
  highlights?: string[];
}

const work: Position[] = [
  {
    name: 'Zhejiang University & University of Illinois Urbana-Champaign',
    position: 'Undergraduate Student',
    url: 'https://illinois.edu',
    startDate: '2024-09-01',
    summary:
      'Pursuing a B.S. in Electrical Engineering, with coursework and research interests in machine learning, embodied AI, and generative models.',
  },
];

export default work;
