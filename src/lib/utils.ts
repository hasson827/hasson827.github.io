/**
 * Shared utility functions and constants
 */

// Site configuration
export const SITE_URL = 'https://hasson827.github.io';
export const AUTHOR_NAME = 'Hongshuo Zhao';
export const TWITTER_HANDLE = undefined;
export const SITE_IMAGE_PATH = '/images/me.jpg';
export const SITE_IMAGE_DIMENSIONS = {
  width: 1280,
  height: 1280,
} as const;

// Canonical one-line bio, shared across page metadata, OpenGraph, and JSON-LD.
export const SITE_DESCRIPTION =
  'An undergraduate student in Electrical Engineering at Zhejiang University and University of Illinois Urbana-Champaign, interested in artificial intelligence, embodied AI, generative models (diffusion models, flow models), and their applications.';

// Image dimension constants
export const AVATAR_SIZE = {
  hero: 120,
  footer: 80,
  sidebar: 200,
} as const;

export const PROJECT_IMAGE = {
  width: 600,
  height: 400,
} as const;

// Skill competency
export const MAX_COMPETENCY = 5;

/**
 * Formats a date string to a human-readable format.
 * Parses as UTC to avoid timezone shifts.
 */
export function formatDate(dateStr: string): string {
  if (!dateStr) return '';
  // Parse as UTC to avoid timezone shifts
  const date = new Date(`${dateStr}T12:00:00`);
  return date.toLocaleDateString('en-US', {
    year: 'numeric',
    month: 'long',
    day: 'numeric',
  });
}
