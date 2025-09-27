import * as http from "http";
import * as https from "https";

const TARGET_URL = process.env.CHECK_URL || "http://localhost:8080/health";
const EXPECT_KEY = process.env.CHECK_EXPECT_KEY || "ok";

function request(url: string): Promise<any> {
  return new Promise((resolve, reject) => {
    const client = url.startsWith('https') ? https : http;
    const req = client.get(url, (res) => {
      const chunks: Buffer[] = [];
      res.on('data', (chunk) => chunks.push(Buffer.from(chunk)));
      res.on('end', () => {
        try {
          const body = Buffer.concat(chunks).toString('utf-8');
          const json = JSON.parse(body);
          resolve(json);
        } catch (err) {
          reject(err);
        }
      });
    });
    req.on('error', reject);
    req.setTimeout(Number(process.env.CHECK_TIMEOUT || 5000), () => {
      req.destroy(new Error('timeout'));
    });
  });
}

async function main(): Promise<void> {
  try {
    const json = await request(TARGET_URL);
    if (!(EXPECT_KEY in json)) {
      console.error(`[Check] Key ${EXPECT_KEY} missing in response.`);
      process.exitCode = 1;
      return;
    }
    console.log(`[Check] ${TARGET_URL} healthy. ${EXPECT_KEY}=${json[EXPECT_KEY]}`);
  } catch (err) {
    console.error('[Check] Inference health check failed:', err);
    process.exitCode = 1;
  }
}

main();
