import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { createClient, parseCliConfig, type Env } from "./config.js";
import { createServer } from "./server.js";

interface ProcessRuntime {
  argv: string[];
  env: Record<string, string | undefined>;
  on: (event: "SIGINT" | "SIGTERM", listener: () => void | Promise<void>) => void;
  exit: (code?: number) => never;
}

const processRuntime = (globalThis as typeof globalThis & {
  process?: ProcessRuntime;
}).process;

if (!processRuntime) {
  throw new Error("The stdio MCP server entrypoint requires a process runtime.");
}

const runtime = processRuntime;

const env: Env = {
  HONCHO_API_URL: runtime.env.HONCHO_API_URL,
};

async function main() {
  const config = parseCliConfig(runtime.argv.slice(2), env);
  const honcho = createClient(config);
  const server = createServer({ honcho, config });
  const transport = new StdioServerTransport();

  transport.onerror = (error) => {
    console.error("MCP stdio transport error:", error);
  };

  await server.connect(transport);

  for (const signal of ["SIGINT", "SIGTERM"] as const) {
    runtime.on(signal, async () => {
      await server.close();
      runtime.exit(0);
    });
  }
}

main().catch((error) => {
  console.error("Failed to start Honcho MCP stdio server:", error);
  runtime.exit(1);
});
