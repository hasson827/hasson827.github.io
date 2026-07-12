import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';

import EmailLink from '../../Contact/EmailLink';

describe('EmailLink', () => {
  it('renders the email address', () => {
    render(<EmailLink />);

    expect(screen.getByText('hasson827624@gmail.com')).toBeInTheDocument();
  });

  it('renders as a link element', () => {
    render(<EmailLink />);

    const link = screen.getByRole('link');
    expect(link).toBeInTheDocument();
  });

  it('uses a mailto href', () => {
    render(<EmailLink />);

    const link = screen.getByRole('link');
    expect(link.getAttribute('href')).toBe('mailto:hasson827624@gmail.com');
  });
});
