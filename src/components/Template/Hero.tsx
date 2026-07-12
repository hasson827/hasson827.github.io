import Link from 'next/link';

import ThemePortrait from './ThemePortrait';

export default function Hero() {
  return (
    <section className="hero">
      <div className="hero-content">
        <div className="hero-avatar">
          <ThemePortrait width={160} height={160} priority />
        </div>

        <h1 className="hero-title">
          <span className="hero-name">Hongshuo Zhao</span>
        </h1>

        <p className="hero-tagline">
          An undergraduate student in Electrical Engineering at{' '}
          <a href="https://www.zju.edu.cn/english/" className="hero-highlight">
            Zhejiang University
          </a>{' '}
          and{' '}
          <a href="https://illinois.edu/" className="hero-highlight">
            University of Illinois Urbana-Champaign
          </a>
          .<br />
          Interested in AI, embodied intelligence, generative models, and their
          applications.
        </p>

        <div className="hero-chips">
          <span className="hero-chip">Zhejiang University</span>
          <span className="hero-chip">UIUC</span>
          <span className="hero-chip">Electrical Engineering</span>
        </div>

        <div className="hero-cta">
          <Link href="/about" className="button">
            About Me
          </Link>
          <Link href="/resume" className="button button-secondary">
            View Resume
          </Link>
        </div>
      </div>

      <div className="hero-bg" aria-hidden="true">
        <div className="hero-gradient" />
      </div>
    </section>
  );
}
