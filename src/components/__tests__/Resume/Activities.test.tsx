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
