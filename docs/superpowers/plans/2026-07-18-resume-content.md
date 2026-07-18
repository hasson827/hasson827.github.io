# Resume Content Update Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Populate the `/resume` page with Hongshuo Zhao's real content from `Resume.tex` (Experience, Education details, new Activities section, Skills, 18 UIUC Courses, header/metadata), replacing all remaining template content.

**Architecture:** Content lives in typed data files under `src/data/resume/` consumed by presentational components in `src/components/Resume/`; the page composes sections in `app/resume/page.tsx` with anchor nav in `ResumeNav.tsx`. One new data file + one new section component (Activities, reusing the existing `Job` card), one small rendering fix (`Job.tsx` same-month dates), one optional field added to `Degree`.

**Tech Stack:** Next.js 16 (App Router, static export), React 19, TypeScript strict, Vitest + Testing Library, Biome + Prettier, dayjs.

**Spec:** `docs/superpowers/specs/2026-07-18-resume-content-design.md` (approved)

## Global Constraints

- Branch: `feat/resume-content` (already created and pushed). Never commit to `main`.
- Run `npm run format` before every commit (CI enforces formatting).
- Push to origin after every commit (`git push`), per AGENTS.md git workflow.
- Conventional commit messages (`feat:`, `fix:`, `test:`, `chore:`).
- No new dependencies.
- Copy (dates, GPAs, percentages, names) must match the spec/`Resume.tex` exactly — including `TOEFL: 5.5` displayed as-is.
- `npm test` runs Vitest in run mode; filter with `npm test -- <name-substring>`.

---

### Task 1: Experience data (`work.ts`) + data test update

**Files:**
- Modify: `src/data/resume/work.ts` (full replacement)
- Test: `src/data/__tests__/work.test.ts` (remove one test block)

**Interfaces:**
- Consumes: nothing new.
- Produces: unchanged `Position` interface and default export `work: Position[]` (Tasks 3, 4 reuse `Position`; page already consumes `work`).

- [ ] **Step 1: Remove the current-position assertion from `work.test.ts`**

Delete this block (lines 59–63):

```ts
  // Resume should show at least one current/active position
  it('has at least one current position (no endDate)', () => {
    const currentJobs = work.filter((job) => !job.endDate);
    expect(currentJobs.length).toBeGreaterThanOrEqual(1);
  });
```

Reason: all four real Experience entries have ended per `Resume.tex`; the invariant was template-derived.

- [ ] **Step 2: Replace `src/data/resume/work.ts` content**

```ts
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
```

- [ ] **Step 3: Run tests, verify pass**

Run: `npm test -- work`
Expected: PASS (all remaining `work data` tests)

- [ ] **Step 4: Format, commit, push**

```bash
npm run format
git add src/data/resume/work.ts src/data/__tests__/work.test.ts
git commit -m "feat: populate resume experience with research and work entries"
git push
```

---

### Task 2: Education details (`degrees.ts` + `Degree.tsx`)

**Files:**
- Modify: `src/data/resume/degrees.ts` (full replacement)
- Modify: `src/components/Resume/Education/Degree.tsx` (full replacement)
- Test: `src/components/__tests__/Resume/Education.test.tsx` (add one test)
- Test: `src/data/__tests__/degrees.test.ts` (add one test)

**Interfaces:**
- Consumes: existing `Degree` type import in `Degree.tsx` and `Education.tsx`.
- Produces: `Degree` gains optional `details?: string[]`; `Degree.tsx` renders `details` as a `ul.points` list under the school line. No consumer changes required (field is optional).

- [ ] **Step 1: Write the failing component test**

In `src/components/__tests__/Resume/Education.test.tsx`, add inside the `describe('Degree', ...)` block (after the `'displays year'` test):

```tsx
  it('renders details when provided', () => {
    const degreeWithDetails = {
      ...mockDegree,
      details: ['GPA: 4.00/4.00', "Dean's List 2024–2025"],
    };

    render(<Degree data={degreeWithDetails} />);

    expect(screen.getByText('GPA: 4.00/4.00')).toBeInTheDocument();
    expect(screen.getByText("Dean's List 2024–2025")).toBeInTheDocument();
  });
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npm test -- Education`
Expected: FAIL — `Unable to find an element with the text: GPA: 4.00/4.00`

- [ ] **Step 3: Add the optional field and populate `degrees.ts`**

Full replacement of `src/data/resume/degrees.ts`:

```ts
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
```

- [ ] **Step 4: Render details in `Degree.tsx`**

Full replacement of `src/components/Resume/Education/Degree.tsx`:

```tsx
import type { Degree as DegreeType } from '@/data/resume/degrees';

interface DegreeProps {
  data: DegreeType;
}

export default function Degree({ data }: DegreeProps) {
  return (
    <article className="degree-container">
      <header>
        <h4 className="degree">{data.degree}</h4>
        <p className="school">
          <a href={data.link}>{data.school}</a>,{' '}
          <time dateTime={String(data.year)}>{data.year}</time>
        </p>
        {data.details ? (
          <ul className="points">
            {data.details.map((detail) => (
              <li key={detail}>{detail}</li>
            ))}
          </ul>
        ) : null}
      </header>
    </article>
  );
}
```

- [ ] **Step 5: Add a data validator to `degrees.test.ts`**

Add at the end of the `describe('degrees data', ...)` block:

```ts
  it('details are non-empty strings when present', () => {
    for (const degree of degrees) {
      if (degree.details) {
        expect(Array.isArray(degree.details)).toBe(true);
        for (const detail of degree.details) {
          expect(typeof detail).toBe('string');
          expect(detail.trim().length).toBeGreaterThan(0);
        }
      }
    }
  });
```

- [ ] **Step 6: Run tests, verify pass**

Run: `npm test -- Education` and `npm test -- degrees`
Expected: PASS (both suites, including the new tests)

- [ ] **Step 7: Format, commit, push**

```bash
npm run format
git add src/data/resume/degrees.ts src/components/Resume/Education/Degree.tsx src/components/__tests__/Resume/Education.test.tsx src/data/__tests__/degrees.test.ts
git commit -m "feat: show GPA and honors in resume education entries"
git push
```

---

### Task 3: Activities data + component

**Files:**
- Create: `src/data/resume/activities.ts`
- Test: `src/data/__tests__/activities.test.ts` (create)
- Create: `src/components/Resume/Activities.tsx`
- Test: `src/components/__tests__/Resume/Activities.test.tsx` (create)

**Interfaces:**
- Consumes: `Position` type from `src/data/resume/work.ts` (Task 1); `Job` card component from `src/components/Resume/Experience/Job`.
- Produces: default export `activities: Position[]` from `@/data/resume/activities`; default export `Activities({ data: Position[] })` from `@/components/Resume/Activities` — consumed by Task 7 (page wiring). Section anchor id: `activities`.

- [ ] **Step 1: Write the failing data test**

Create `src/data/__tests__/activities.test.ts`:

```ts
import { describe, expect, it } from 'vitest';

import activities from '../resume/activities';

describe('activities data', () => {
  it('exports an array of activities', () => {
    expect(Array.isArray(activities)).toBe(true);
    expect(activities.length).toBeGreaterThan(0);
  });

  it('each activity has required properties', () => {
    for (const activity of activities) {
      expect(activity).toHaveProperty('name');
      expect(activity).toHaveProperty('position');
      expect(activity).toHaveProperty('url');
      expect(activity).toHaveProperty('startDate');

      expect(typeof activity.name).toBe('string');
      expect(typeof activity.position).toBe('string');
      expect(typeof activity.url).toBe('string');
      expect(typeof activity.startDate).toBe('string');
    }
  });

  it('dates are valid and endDate is after startDate when present', () => {
    for (const activity of activities) {
      const start = new Date(activity.startDate);
      expect(start.toString()).not.toBe('Invalid Date');
      if (activity.endDate) {
        const end = new Date(activity.endDate);
        expect(end.toString()).not.toBe('Invalid Date');
        expect(end.getTime()).toBeGreaterThan(start.getTime());
      }
    }
  });

  it('urls are valid', () => {
    const urlRegex = /^https?:\/\/.+/;

    for (const activity of activities) {
      expect(activity.url).toMatch(urlRegex);
    }
  });

  it('highlights are non-empty arrays when present', () => {
    for (const activity of activities) {
      if (activity.highlights) {
        expect(Array.isArray(activity.highlights)).toBe(true);
        expect(activity.highlights.length).toBeGreaterThan(0);
      }
    }
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npm test -- activities`
Expected: FAIL — `Cannot find module '../resume/activities'`

- [ ] **Step 3: Create `src/data/resume/activities.ts`**

```ts
import type { Position } from './work';

const activities: Position[] = [
  {
    name: 'ZJU International Campus Male Volleyball Team',
    position: 'Captain',
    url: 'https://www.zju.edu.cn/english/',
    startDate: '2025-09-01',
    highlights: [
      'First place in the 2025 iZJU Pioneer Cup Volleyball Competition.',
      'Second place in the 2026 iZJU Pioneer Cup Volleyball Competition.',
    ],
  },
  {
    name: 'ZJU Outreach Delegation to Suzhou',
    position: 'Team Leader',
    url: 'https://www.zju.edu.cn/english/',
    startDate: '2025-01-01',
    endDate: '2026-01-31',
    highlights: [
      'Seasonal activity: January 2025 and January 2026.',
      "Planned and executed return-to-school promotional campaigns and managed team logistics, enhancing the university's regional visibility.",
    ],
  },
  {
    name: 'Teaching Team at Juxi Primary School',
    position: 'Vice-Captain',
    url: 'https://www.zju.edu.cn/english/',
    startDate: '2025-07-01',
    endDate: '2025-07-31',
    highlights: [
      'Conducted a 21-day educational program, designed engaging curricula, and provided academic support to rural students.',
    ],
  },
];

export default activities;
```

- [ ] **Step 4: Run data test, verify pass**

Run: `npm test -- activities`
Expected: PASS

- [ ] **Step 5: Write the failing component test**

Create `src/components/__tests__/Resume/Activities.test.tsx`:

```tsx
import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';

import Activities from '../../Resume/Activities';

const mockActivities = [
  {
    name: 'Volleyball Team',
    position: 'Captain',
    url: 'https://example.com',
    startDate: '2025-09-01',
    highlights: ['Won the championship'],
  },
  {
    name: 'Outreach Delegation',
    position: 'Team Leader',
    url: 'https://example.org',
    startDate: '2025-01-01',
    endDate: '2026-01-31',
    highlights: ['Ran promotional campaigns'],
  },
];

describe('Activities', () => {
  it('renders the activities section with title', () => {
    render(<Activities data={mockActivities} />);

    expect(
      screen.getByRole('heading', { name: /activities/i }),
    ).toBeInTheDocument();
  });

  it('renders all activities', () => {
    render(<Activities data={mockActivities} />);

    expect(screen.getByText('Volleyball Team')).toBeInTheDocument();
    expect(screen.getByText('Outreach Delegation')).toBeInTheDocument();
  });

  it('renders activity positions', () => {
    render(<Activities data={mockActivities} />);

    expect(screen.getByText(/Captain/)).toBeInTheDocument();
    expect(screen.getByText(/Team Leader/)).toBeInTheDocument();
  });

  it('has anchor link for navigation', () => {
    render(<Activities data={mockActivities} />);

    const anchor = document.getElementById('activities');
    expect(anchor).toBeInTheDocument();
  });

  it('handles empty activities array', () => {
    render(<Activities data={[]} />);

    expect(
      screen.getByRole('heading', { name: /activities/i }),
    ).toBeInTheDocument();
    const articles = document.querySelectorAll('.jobs-container');
    expect(articles.length).toBe(0);
  });
});
```

- [ ] **Step 6: Run test to verify it fails**

Run: `npm test -- Activities`
Expected: FAIL — `Cannot find module '../../Resume/Activities'`

- [ ] **Step 7: Create `src/components/Resume/Activities.tsx`**

```tsx
import type { Position } from '@/data/resume/work';

import Job from './Experience/Job';

interface ActivitiesProps {
  data: Position[];
}

export default function Activities({ data }: ActivitiesProps) {
  return (
    <div className="activities">
      <div className="link-to" id="activities" />
      <div className="title">
        <h3>Activities</h3>
      </div>
      {data.map((activity) => (
        <Job data={activity} key={`${activity.name}-${activity.position}`} />
      ))}
    </div>
  );
}
```

- [ ] **Step 8: Run tests, verify pass**

Run: `npm test -- Activities`
Expected: PASS (both the data test file and the component test file match this filter)

- [ ] **Step 9: Format, commit, push**

```bash
npm run format
git add src/data/resume/activities.ts src/components/Resume/Activities.tsx src/data/__tests__/activities.test.ts src/components/__tests__/Resume/Activities.test.tsx
git commit -m "feat: add activities section data and component"
git push
```

---

### Task 4: Same-month date rendering fix (`Job.tsx`)

**Files:**
- Modify: `src/components/Resume/Experience/Job.tsx`
- Test: `src/components/__tests__/Resume/Job.test.tsx` (add one test)

**Interfaces:**
- Consumes: unchanged `Position` type.
- Produces: unchanged `Job` props. Rendering change only: when `startDate` and `endDate` fall in the same calendar month, the date range renders the month once instead of twice. Needed by the Juxi entry (July 2025) from Task 3; no consumer changes.

- [ ] **Step 1: Write the failing test**

In `src/components/__tests__/Resume/Job.test.tsx`, add after the `'shows Present for current job (no end date)'` test:

```tsx
  it('renders a single date when start and end are in the same month', () => {
    const oneMonthJob = {
      ...mockJob,
      startDate: '2025-07-01',
      endDate: '2025-07-31',
    };

    render(<Job data={oneMonthJob} />);

    const matches = screen.getAllByText(/july 2025/i);
    expect(matches.length).toBe(1);
  });
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npm test -- Job`
Expected: FAIL — the new test finds 2 elements matching `/july 2025/i`

- [ ] **Step 3: Implement the fix in `Job.tsx`**

Full replacement of `src/components/Resume/Experience/Job.tsx`:

```tsx
import dayjs from 'dayjs';

import type { Position } from '@/data/resume/work';

import JobSummary from './JobSummary';

interface JobProps {
  data: Position;
}

export default function Job({ data }: JobProps) {
  const { name, position, url, startDate, endDate, summary, highlights } = data;

  const sameMonth =
    endDate !== undefined && dayjs(startDate).isSame(dayjs(endDate), 'month');

  return (
    <article className="jobs-container">
      <header>
        <h4>
          <a href={url}>{name}</a> - {position}
        </h4>
        <p className="daterange">
          {' '}
          <time dateTime={startDate}>
            {dayjs(startDate).format('MMMM YYYY')}
          </time>{' '}
          {sameMonth ? null : (
            <>
              -{' '}
              {endDate ? (
                <time dateTime={endDate}>
                  {dayjs(endDate).format('MMMM YYYY')}
                </time>
              ) : (
                'Present'
              )}
            </>
          )}
        </p>
      </header>
      {summary ? <JobSummary summary={summary} /> : null}
      {highlights ? (
        <ul className="points">
          {highlights.map((highlight) => (
            <li key={highlight}>{highlight}</li>
          ))}
        </ul>
      ) : null}
    </article>
  );
}
```

Note: `endDate !== undefined` keeps the `endDate ?` narrowing inside the fragment intact for TypeScript; `dayjs.isSame(other, 'month')` is core dayjs API (no plugin).

- [ ] **Step 4: Run tests, verify pass**

Run: `npm test -- Job`
Expected: PASS (including pre-existing date-range and Present tests)

- [ ] **Step 5: Format, commit, push**

```bash
npm run format
git add src/components/Resume/Experience/Job.tsx src/components/__tests__/Resume/Job.test.tsx
git commit -m "fix: render single date for same-month date ranges in resume jobs"
git push
```

---

### Task 5: Skills data (`skills.ts`)

**Files:**
- Modify: `src/data/resume/skills.ts` (replace the `skills` array only)

**Interfaces:**
- Consumes: nothing new.
- Produces: unchanged `Skill`/`Category` interfaces and `skills`/`categories` exports. Categories become `English`, `Languages`, `Theory` (auto-built, alphabetical).

- [ ] **Step 1: Replace the `skills` array in `src/data/resume/skills.ts`**

Keep the `Skill`/`Category` interfaces, the `.map(...)` sort, `buildCategories`, and the exports exactly as they are. Replace only the array literal:

```ts
const skills: Skill[] = [
  // Languages
  {
    title: 'C++',
    competency: 5,
    category: ['Languages'],
  },
  {
    title: 'Python',
    competency: 5,
    category: ['Languages'],
  },
  {
    title: 'MATLAB',
    competency: 4,
    category: ['Languages'],
  },
  {
    title: 'LaTeX',
    competency: 4,
    category: ['Languages'],
  },
  {
    title: 'Markdown',
    competency: 4,
    category: ['Languages'],
  },
  // Theory
  {
    title: 'Machine Learning',
    competency: 4,
    category: ['Theory'],
  },
  {
    title: 'Generative Modeling',
    competency: 4,
    category: ['Theory'],
  },
  {
    title: 'Data Structures',
    competency: 4,
    category: ['Theory'],
  },
  {
    title: 'Algorithms',
    competency: 4,
    category: ['Theory'],
  },
  // English
  {
    title: 'CET-4: 624',
    competency: 3,
    category: ['English'],
  },
  {
    title: 'CET-6: 597',
    competency: 3,
    category: ['English'],
  },
  {
    title: 'TOEFL: 5.5',
    competency: 3,
    category: ['English'],
  },
].map((skill) => ({ ...skill, category: skill.category.sort() }));
```

(Competency only controls sort order in the tag cloud; it is not displayed.)

- [ ] **Step 2: Run tests, verify pass**

Run: `npm test -- skills`
Expected: PASS (`skills data` and `categories data` suites, plus component `Skills`/`SkillTag`/`CategoryButton` suites that use their own mocks)

- [ ] **Step 3: Format, commit, push**

```bash
npm run format
git add src/data/resume/skills.ts
git commit -m "feat: replace resume skills with personal languages, theory, and english scores"
git push
```

---

### Task 6: Courses data (`courses.ts`)

**Files:**
- Modify: `src/data/resume/courses.ts` (full replacement)

**Interfaces:**
- Consumes: nothing new.
- Produces: unchanged `Course` interface and default export `courses: Course[]`. All entries use `university: 'UIUC'` and links of the form `https://courses.illinois.edu/schedule/terms/{DEPT}/{NUM}`.

- [ ] **Step 1: Replace `src/data/resume/courses.ts` content**

```ts
export interface Course {
  title: string;
  number: string;
  link: string;
  university: string;
}

const courses: Course[] = [
  {
    title: 'Introduction to Electronics',
    number: 'ECE 110',
    link: 'https://courses.illinois.edu/schedule/terms/ECE/110',
    university: 'UIUC',
  },
  {
    title: 'Introduction to Computing',
    number: 'ECE 120',
    link: 'https://courses.illinois.edu/schedule/terms/ECE/120',
    university: 'UIUC',
  },
  {
    title: 'Analog Signal Processing',
    number: 'ECE 210',
    link: 'https://courses.illinois.edu/schedule/terms/ECE/210',
    university: 'UIUC',
  },
  {
    title: 'Computer Systems & Programming',
    number: 'ECE 220',
    link: 'https://courses.illinois.edu/schedule/terms/ECE/220',
    university: 'UIUC',
  },
  {
    title: 'Calculus I',
    number: 'MATH 221',
    link: 'https://courses.illinois.edu/schedule/terms/MATH/221',
    university: 'UIUC',
  },
  {
    title: 'Calculus II',
    number: 'MATH 231',
    link: 'https://courses.illinois.edu/schedule/terms/MATH/231',
    university: 'UIUC',
  },
  {
    title: 'Calculus III',
    number: 'MATH 241',
    link: 'https://courses.illinois.edu/schedule/terms/MATH/241',
    university: 'UIUC',
  },
  {
    title: 'Intro Differential Equations',
    number: 'MATH 285',
    link: 'https://courses.illinois.edu/schedule/terms/MATH/285',
    university: 'UIUC',
  },
  {
    title: 'Basic Discrete Mathematics',
    number: 'MATH 213',
    link: 'https://courses.illinois.edu/schedule/terms/MATH/213',
    university: 'UIUC',
  },
  {
    title: 'Linear Algebra with Computational Applications',
    number: 'MATH 257',
    link: 'https://courses.illinois.edu/schedule/terms/MATH/257',
    university: 'UIUC',
  },
  {
    title: 'Intro Computing: Engrg & Sci',
    number: 'CS 101',
    link: 'https://courses.illinois.edu/schedule/terms/CS/101',
    university: 'UIUC',
  },
  {
    title: 'Data Structures',
    number: 'CS 225',
    link: 'https://courses.illinois.edu/schedule/terms/CS/225',
    university: 'UIUC',
  },
  {
    title: 'Principles of Writing',
    number: 'RHET 101',
    link: 'https://courses.illinois.edu/schedule/terms/RHET/101',
    university: 'UIUC',
  },
  {
    title: 'Principles of Research',
    number: 'RHET 102',
    link: 'https://courses.illinois.edu/schedule/terms/RHET/102',
    university: 'UIUC',
  },
  {
    title: 'University Physics: Mechanics',
    number: 'PHYS 211',
    link: 'https://courses.illinois.edu/schedule/terms/PHYS/211',
    university: 'UIUC',
  },
  {
    title: 'University Physics: Elec & Mag',
    number: 'PHYS 212',
    link: 'https://courses.illinois.edu/schedule/terms/PHYS/212',
    university: 'UIUC',
  },
  {
    title: 'Univ Physics: Thermal Physics',
    number: 'PHYS 213',
    link: 'https://courses.illinois.edu/schedule/terms/PHYS/213',
    university: 'UIUC',
  },
  {
    title: 'Univ Physics: Quantum Physics',
    number: 'PHYS 214',
    link: 'https://courses.illinois.edu/schedule/terms/PHYS/214',
    university: 'UIUC',
  },
];

export default courses;
```

(Titles verified against catalog.illinois.edu on 2026-07-18; the `courses.illinois.edu/schedule/terms/...` URL pattern is the official per-course catalog link. Note it 403s non-browser fetchers — fine as an href.)

- [ ] **Step 2: Run tests, verify pass**

Run: `npm test -- courses`
Expected: PASS (`courses data` suite; component `Courses` suite uses its own mocks)

- [ ] **Step 3: Format, commit, push**

```bash
npm run format
git add src/data/resume/courses.ts
git commit -m "feat: replace template courses with UIUC coursework"
git push
```

---

### Task 7: Page wiring (`page.tsx` + `ResumeNav.tsx`) + nav test

**Files:**
- Modify: `app/resume/page.tsx` (full replacement)
- Modify: `src/components/Resume/ResumeNav.tsx` (one-line change to `sections` array)
- Test: `src/components/__tests__/Resume/ResumeNav.test.tsx` (update two tests)

**Interfaces:**
- Consumes: `activities` from `@/data/resume/activities` and `Activities` from `@/components/Resume/Activities` (Task 3).
- Produces: `/resume` page with section order Experience → Education → Activities → Skills → Courses → References; nav entry `{ name: 'Activities', id: 'activities' }` between Education and Skills.

- [ ] **Step 1: Update the failing nav tests**

In `src/components/__tests__/Resume/ResumeNav.test.tsx`:

1. In the `'renders links to all resume sections'` test, add after the `#education` assertion:

```tsx
    expect(
      screen.getByRole('link', { name: /activities/i }),
    ).toHaveAttribute('href', '#activities');
```

2. Replace the `'renders 5 navigation links'` test with:

```tsx
  it('renders 6 navigation links', () => {
    render(<ResumeNav />);

    const links = screen.getAllByRole('link');
    expect(links.length).toBe(6);
  });
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npm test -- ResumeNav`
Expected: FAIL — `Unable to find an accessible element with the role "link" and name /activities/i`

- [ ] **Step 3: Add the Activities entry in `ResumeNav.tsx`**

Change the `sections` array to:

```ts
const sections = [
  { name: 'Experience', id: 'experience' },
  { name: 'Education', id: 'education' },
  { name: 'Activities', id: 'activities' },
  { name: 'Skills', id: 'skills' },
  { name: 'Courses', id: 'courses' },
  { name: 'References', id: 'references' },
] as const;
```

- [ ] **Step 4: Update `app/resume/page.tsx`**

Full replacement:

```tsx
import type { Metadata } from 'next';

import Activities from '@/components/Resume/Activities';
import Courses from '@/components/Resume/Courses';
import Education from '@/components/Resume/Education';
import Experience from '@/components/Resume/Experience';
import References from '@/components/Resume/References';
import ResumeNav from '@/components/Resume/ResumeNav';
import Skills from '@/components/Resume/Skills';
import PageWrapper from '@/components/Template/PageWrapper';
import activities from '@/data/resume/activities';
import courses from '@/data/resume/courses';
import degrees from '@/data/resume/degrees';
import { categories, skills } from '@/data/resume/skills';
import work from '@/data/resume/work';
import { createPageMetadata } from '@/lib/metadata';

export const metadata: Metadata = createPageMetadata({
  title: 'Resume',
  description:
    "Hongshuo Zhao's resume — ZJU–UIUC Electrical Engineering dual-degree student researching machine learning for materials science and diffusion-based multimodal LLMs.",
  path: '/resume/',
});

export default function ResumePage() {
  return (
    <PageWrapper>
      <section className="resume-page">
        <header className="resume-header">
          <h1 className="resume-title">Resume</h1>
          <p className="resume-summary">
            Electrical Engineering undergraduate in the ZJU–UIUC dual-degree
            program. My research spans machine learning for materials science
            and efficient inference for diffusion-based multimodal LLMs.
            Ranked 1/71 at Zhejiang University with a 4.00 GPA at UIUC.
          </p>
        </header>

        <ResumeNav />

        <div className="resume-content">
          <section id="experience" className="resume-section">
            <Experience data={work} />
          </section>

          <section id="education" className="resume-section">
            <Education data={degrees} />
          </section>

          <section id="activities" className="resume-section">
            <Activities data={activities} />
          </section>

          <section id="skills" className="resume-section">
            <Skills skills={skills} categories={categories} />
          </section>

          <section id="courses" className="resume-section">
            <Courses data={courses} />
          </section>

          <section id="references" className="resume-section">
            <References />
          </section>
        </div>
      </section>
    </PageWrapper>
  );
}
```

- [ ] **Step 5: Run tests, verify pass**

Run: `npm test -- ResumeNav`
Expected: PASS (6 links, activities link present)

- [ ] **Step 6: Format, commit, push**

```bash
npm run format
git add app/resume/page.tsx src/components/Resume/ResumeNav.tsx src/components/__tests__/Resume/ResumeNav.test.tsx
git commit -m "feat: wire activities section into resume page and update header copy"
git push
```

---

### Task 8: Full verification

**Files:** none (verification only; commit formatting fallout if any).

- [ ] **Step 1: Format check**

Run: `npm run format && git status --porcelain`
Expected: no output from `git status --porcelain` (working tree clean apart from untracked pre-existing files like `Resume.pdf`). If files changed, `git add -A` the changed files, commit `chore: format resume content files`, push.

- [ ] **Step 2: Lint**

Run: `npm run lint`
Expected: no errors

- [ ] **Step 3: Type check**

Run: `npm run type-check`
Expected: no errors

- [ ] **Step 4: Full test suite**

Run: `npm test`
Expected: all suites PASS

- [ ] **Step 5: Production build (static export)**

Run: `npm run build`
Expected: build succeeds; `out/resume/index.html` is emitted

- [ ] **Step 6: Content smoke check on the exported page**

Run:

```bash
grep -c "VRCD" out/resume/index.html
grep -c "Rank 1/71" out/resume/index.html
grep -c "TOEFL: 5.5" out/resume/index.html
grep -c "Univ Physics: Quantum Physics" out/resume/index.html
grep -c "Juxi Primary School" out/resume/index.html
```

Expected: each prints a count ≥ 1

- [ ] **Step 7: Push (if any commit was made in Step 1)**

```bash
git push
```

---

## Self-Review Notes

- **Spec coverage:** §1 Experience → Task 1; §2 Education → Task 2; §3 Activities → Tasks 3+7 (+Job fix Task 4); §4 Skills → Task 5; §5 Courses → Task 6; §6 header/metadata → Task 7; §7 tests → Tasks 1–4, 7; verification → Task 8. Out-of-scope items (References, Resume.pdf, ZJU courses) untouched.
- **Placeholders:** none — all steps carry complete code/commands.
- **Type consistency:** `Position` reused by `activities.ts`/`Activities.tsx`/`Job` (Tasks 1, 3, 4, 7); `Degree.details?: string[]` (Task 2) matches its rendering and tests; nav `sections` shape unchanged (Task 7).
