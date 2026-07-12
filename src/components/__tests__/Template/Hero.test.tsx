import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';

import Hero from '../../Template/Hero';

describe('Hero', () => {
  it('renders the hero section', () => {
    render(<Hero />);

    const heroSection = document.querySelector('.hero');
    expect(heroSection).toBeInTheDocument();
  });

  it('displays the name as heading', () => {
    render(<Hero />);

    const heading = screen.getByRole('heading', { level: 1 });
    expect(heading).toHaveTextContent('Hongshuo Zhao');
  });

  it('renders the tagline with university links', () => {
    render(<Hero />);

    const zjuLink = screen.getByRole('link', { name: /zhejiang university/i });
    expect(zjuLink).toHaveAttribute('href', 'https://www.zju.edu.cn/english/');
    expect(zjuLink).toHaveClass('hero-highlight');

    const uiucLink = screen.getByRole('link', {
      name: /university of illinois urbana-champaign/i,
    });
    expect(uiucLink).toHaveAttribute('href', 'https://illinois.edu/');
    expect(uiucLink).toHaveClass('hero-highlight');
  });

  it('displays hero chips for credentials', () => {
    render(<Hero />);

    const chips = document.querySelector('.hero-chips');
    expect(chips).toBeInTheDocument();
    expect(chips).toHaveTextContent('Zhejiang University');
    expect(chips).toHaveTextContent('UIUC');
    expect(chips).toHaveTextContent('Electrical Engineering');
  });

  it('renders CTA buttons with correct links', () => {
    render(<Hero />);

    const aboutButton = screen.getByRole('link', { name: /about me/i });
    expect(aboutButton).toHaveAttribute('href', '/about');
    expect(aboutButton).toHaveClass('button');

    const resumeButton = screen.getByRole('link', { name: /view resume/i });
    expect(resumeButton).toHaveAttribute('href', '/resume');
    expect(resumeButton).toHaveClass('button-secondary');
  });

  it('has decorative background elements', () => {
    render(<Hero />);

    const bg = document.querySelector('.hero-bg');
    expect(bg).toBeInTheDocument();
    expect(bg).toHaveAttribute('aria-hidden', 'true');
  });
});
