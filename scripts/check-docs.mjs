import { existsSync, readFileSync } from 'node:fs';

const required = [
  'README.md',
  'docs/adr/0001-use-microwakeword-and-local-tts.md',
  'docs/adr/0002-human-voice-data-privacy-boundary.md',
  'docs/design/training-pipeline.md',
  'docs/design/web-voice-collection.md',
  'docs/plans/2026-05-11-wakeword-pipeline-plan.md',
];

const missing = required.filter((path) => !existsSync(path));
if (missing.length > 0) {
  console.error(`Missing required docs:\n${missing.map((p) => `- ${p}`).join('\n')}`);
  process.exit(1);
}

const readme = readFileSync('README.md', 'utf8');
for (const phrase of ['ハイ、ｽﾀｯｸﾁｬﾝ', 'microWakeWord', 'local TTS', 'human voice']) {
  if (!readme.includes(phrase)) {
    console.error(`README.md must mention: ${phrase}`);
    process.exit(1);
  }
}
