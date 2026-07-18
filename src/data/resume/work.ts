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
    name: 'Zhejiang University',
    position:
      'Undergraduate Researcher — VRCD (advised by Prof. Xiangming Meng)',
    url: 'https://www.zju.edu.cn/english/',
    startDate: '2026-03-01',
    endDate: '2026-05-31',
    summary:
      'Visual-Redundancy-Controlled Parallel Decoding for Diffusion-Based Multimodal Large Language Models. Submitted to NeurIPS 2026.',
    highlights: [
      'Proposed the Visual Redundancy Index (VRI), a metric that quantifies visual grounding overlap among textual tokens unmasked in parallel within diffusion-based multimodal LLMs.',
      'Designed VRCD, a training-free inference-time reranking method that leverages token-to-image attention to re-weight confidence scores, prioritizing visually complementary decoding positions.',
      'Achieved relative accuracy gains of up to 18.8% on M3CoT and 6.9% on MMBench over confidence-based decoding, with merely ~1.5% runtime overhead.',
    ],
  },
  {
    name: 'Zhejiang University',
    position:
      'Undergraduate Researcher — Phonon Band Prediction (National Innovation Project, advised by Prof. Wee-Liat Ong)',
    url: 'https://www.zju.edu.cn/english/',
    startDate: '2025-05-01',
    endDate: '2026-05-31',
    summary:
      'Phonon band prediction for screening of material thermal properties.',
    highlights: [
      'Proposed two attention-enhanced GNN schemes (scalar and E(3)-equivariant attention) for crystal phonon band prediction, achieving up to 17.0% relative error reduction while preserving physical symmetries.',
      'Conducted systematic experiments and ablation studies on ~1,500 inorganic crystals, establishing a reproducible ML pipeline for high-throughput thermal property screening.',
    ],
  },
  {
    name: 'Zhejiang University',
    position: 'Undergraduate Teaching Assistant — MATH 257 Linear Algebra',
    url: 'https://www.zju.edu.cn/english/',
    startDate: '2026-03-01',
    endDate: '2026-06-30',
    highlights: [
      'Facilitated discussion sessions and office hours for MATH 257 (Linear Algebra).',
      'Directed and graded course assignments.',
    ],
  },
  {
    name: 'Zhejiang University',
    position: 'Admission Team Member — Suzhou, Jiangsu',
    url: 'https://www.zju.edu.cn/english/',
    startDate: '2025-06-01',
    endDate: '2026-06-30',
    highlights: [
      'Seasonal role covering the June 2025 and June 2026 admissions cycles.',
      "Provided application recommendations for candidates based on previous years' data.",
    ],
  },
];

export default work;
