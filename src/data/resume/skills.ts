export interface Skill {
  title: string;
  competency: number;
  category: string[];
}

export interface Category {
  name: string;
  color: string;
}

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

/**
 * Build categories from skills, all using the accent color token.
 */
function buildCategories(skillsList: Skill[]): Category[] {
  const uniqueCategories = Array.from(
    new Set(skillsList.flatMap(({ category }) => category)),
  ).sort();

  return uniqueCategories.map((category) => ({
    name: category,
    color: 'var(--color-accent)',
  }));
}

const categories: Category[] = buildCategories(skills);

export { categories, skills };
