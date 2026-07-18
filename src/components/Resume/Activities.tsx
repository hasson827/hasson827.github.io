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
