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
