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
