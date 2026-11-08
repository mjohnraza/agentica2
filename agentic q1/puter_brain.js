/**
 * Puter.js LLM Brain implementation for SaafHawa Agent (Node.js runtime).
 * Powered by Puter (https://github.com/heyPuter/puter)
 */

import { init, getAuthToken } from '@heyputer/puter.js/src/init.cjs';
import process from 'node:process';

const token = process.env.PUTER_AUTH_TOKEN;

async function run() {
  console.log('=== SaafHawa Node.js Puter LLM Brain ===');
  let puter;

  if (token) {
    console.log('Initializing Puter with token...');
    puter = init(token);
  } else {
    console.log('No PUTER_AUTH_TOKEN set.');
    console.log('To authenticate interactively, run with: PUTER_AUTH_TOKEN=your_token node puter_brain.js');
    console.log('Or use the Python LLM brain in model.py which supports offline deterministic evaluation.');
    return;
  }

  try {
    const prompt = `Citizen Query: Aj din 12 bajy temperature kitna rise hoga qk bachon ki school 12 bajy off hojayega?
You are SaafHawa agent. Provide guidance on whether to call tools (get_aqi, get_weather) or give advice.`;

    const response = await puter.ai.chat(prompt, {
      model: process.env.PUTER_MODEL || 'gpt-4o-mini',
    });

    console.log('\nPuter Response:');
    console.log(typeof response === 'string' ? response : JSON.stringify(response, null, 2));
  } catch (err) {
    console.error('Puter API Error:', err.message);
  }
}

run();

