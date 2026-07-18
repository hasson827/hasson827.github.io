# Resume Content Update — Design

**Date**: 2026-07-18
**Branch**: `feat/resume-content`
**Source of truth**: `Resume.tex` (user's CV) + user-provided UIUC course list

## Goal

Populate the `/resume` page with Hongshuo Zhao's real content from `Resume.tex`, replacing all remaining template (Michael D'Angelo) content: Experience entries, Education details, Skills, Courses, page header summary, and metadata.

## User-approved decisions

1. Research experience (VRCD, Phonon) goes into the existing **Experience** section — no new Research section.
2. Courses: user provided 18 UIUC course codes; titles verified against the official UIUC catalog (see below).
3. English scores displayed as-is from the .tex, including `TOEFL (5.5)`.
4. Extracurriculars get a **new Activities section** between Education and Skills.
5. Education entries gain optional detail lines (GPA, rank, honors).

## Changes

### 1. `src/data/resume/work.ts` — Experience

Replace the single "Undergraduate Student" placeholder with 4 entries (all `name: 'Zhejiang University'`, `url: 'https://www.zju.edu.cn/english/'`), in this order:

| position | startDate | endDate | highlights |
|---|---|---|---|
| Undergraduate Researcher — VRCD (advised by Prof. Xiangming Meng) | 2026-03-01 | 2026-05-31 | Full title "Visual-Redundancy-Controlled Parallel Decoding for Diffusion-Based Multimodal LLMs" + "Submitted to NeurIPS 2026" in `summary`; 3 bullets from .tex (VRI metric; training-free attention reranking; +18.8% M3CoT / +6.9% MMBench at ~1.5% overhead) |
| Undergraduate Researcher — Phonon Band Prediction (National Innovation Project, advised by Prof. Wee-Liat Ong) | 2025-05-01 | 2026-05-31 | 2 bullets: attention-enhanced GNNs (scalar + E(3)-equivariant), 17.0% relative error reduction; reproducible ML pipeline over ~1,500 inorganic crystals |
| Undergraduate Teaching Assistant — MATH 257 Linear Algebra | 2026-03-01 | 2026-06-30 | discussion sessions / office hours; directed and graded assignments |
| Admission Team Member — Suzhou, Jiangsu | 2025-06-01 | 2026-06-30 | first highlight clarifies "Seasonal role: June 2025 and June 2026 admissions cycles"; second highlight: application recommendations from previous years' data |

### 2. `src/data/resume/degrees.ts` + `src/components/Resume/Education/Degree.tsx`

- Add optional `details?: string[]` to the `Degree` interface; render as a small list under the school line in `Degree.tsx`.
- UIUC: `['GPA: 4.00/4.00', "Dean's List 2024–2025, 2025–2026"]`
- ZJU: `['GPA: 3.93/4.00 — Rank 1/71', 'National Scholarship; ZJU First-Class Scholarship; ZJU First-Class Institute Scholarship']`
- Existing degree order (UIUC first, both year 2028) unchanged.

### 3. Activities (new section)

- New data file `src/data/resume/activities.ts`, reusing the `Position` interface imported from `work.ts`.
- New component `src/components/Resume/Activities.tsx` mirroring `Experience.tsx` (renders `Job` cards).
- Entries:
  1. **Captain**, ZJU International Campus Male Volleyball Team — 2025-09 → Present — highlights: 1st place 2025 iZJU Pioneer Cup; 2nd place 2026 iZJU Pioneer Cup.
  2. **Team Leader**, ZJU Outreach Delegation to Suzhou — 2025-01 → 2026-01 — first highlight "Seasonal: January 2025 and January 2026"; then planned/executed return-to-school campaigns, team logistics, regional visibility.
  3. **Vice-Captain**, Teaching Team at Juxi Primary School — 2025-07 → 2025-07 — 21-day educational program, curriculum design, academic support for rural students.
- `app/resume/page.tsx`: insert `<Activities data={activities} />` section between Education and Skills.
- `ResumeNav.tsx`: add `{ name: 'Activities', id: 'activities' }` after Education.

**Small component fix in `Job.tsx`**: when start and end fall in the same month/year (Juxi, July 2025), render the date once instead of "July 2025 - July 2025".

### 4. `src/data/resume/skills.ts`

Replace all template skills. Categories (auto-built, alphabetical): `English`, `Languages`, `Theory`.

- **Languages**: C++ (5), Python (5), MATLAB (4), Markdown (3), LaTeX (3)
- **Theory**: Machine Learning (4), Data Structures (4), Algorithms (4), Generative Modeling (4)
- **English**: `CET-4: 624` (3), `CET-6: 597` (3), `TOEFL: 5.5` (3)

Competency only controls tag sort order within the cloud; it is not displayed.

### 5. `src/data/resume/courses.ts`

Replace 13 Stanford template courses with the user's 18 UIUC courses. All `university: 'UIUC'`, links use the official catalog pattern `https://courses.illinois.edu/schedule/terms/{DEPT}/{NUM}`. Titles verified at catalog.illinois.edu (2026-07-18):

| number | title |
|---|---|
| ECE 110 | Introduction to Electronics |
| ECE 120 | Introduction to Computing |
| ECE 210 | Analog Signal Processing |
| ECE 220 | Computer Systems & Programming |
| MATH 221 | Calculus I |
| MATH 231 | Calculus II |
| MATH 241 | Calculus III |
| MATH 285 | Intro Differential Equations |
| MATH 213 | Basic Discrete Mathematics |
| MATH 257 | Linear Algebra with Computational Applications |
| CS 101 | Intro Computing: Engrg & Sci |
| CS 225 | Data Structures |
| RHET 101 | Principles of Writing |
| RHET 102 | Principles of Research |
| PHYS 211 | University Physics: Mechanics |
| PHYS 212 | University Physics: Elec & Mag |
| PHYS 213 | Univ Physics: Thermal Physics |
| PHYS 214 | Univ Physics: Quantum Physics |

Note: `courses.illinois.edu` returns 403 to non-browser fetchers but is the official public catalog URL (confirmed from catalog.illinois.edu itself); fine as an href.

### 6. `app/resume/page.tsx` — header + metadata

- `resume-summary` final wording (aligned with `src/data/about.ts` intro and `Hero.tsx`): "Electrical Engineering undergraduate in the ZJU–UIUC dual-degree program. My research spans machine learning for materials science and efficient inference for diffusion-based multimodal LLMs. Ranked 1/71 at Zhejiang University with a 4.00 GPA at UIUC."
- `metadata.description`: "Hongshuo Zhao's resume — ZJU–UIUC Electrical Engineering dual-degree student researching machine learning for materials science and diffusion-based multimodal LLMs."

### 7. Tests

- `src/data/__tests__/work.test.ts`: remove the "has at least one current position (no endDate)" assertion — all four Experience entries have ended per the .tex; the invariant was template-derived.
- New `src/data/__tests__/activities.test.ts`: mirror `work.test.ts` validators (required props, valid dates, valid URLs, endDate > startDate when present, etc.). The volleyball captaincy (no endDate) satisfies any current-entry check.
- `degrees.test.ts`, `skills.test.ts`, `courses.test.ts`: generic validators, expected to pass unchanged with new data.

### Out of scope

- References section stays "available upon request".
- No Resume.pdf download link (untracked `Resume.pdf` in repo root is untouched).
- ZJU courses are not added (user specified UIUC only).

## Verification

`npm run format`, `npm run lint`, `npm run type-check`, `npm test`, plus `npm run build` (static export) and visual check of `/resume` in dev server.
