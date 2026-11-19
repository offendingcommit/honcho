import { isInitializeRequest } from "@modelcontextprotocol/sdk/types.js";
import { WebStandardStreamableHTTPServerTransport } from "@modelcontextprotocol/sdk/server/webStandardStreamableHttp.js";
import { createClient, parseConfig, type Env } from "./config.js";
import { createServer } from "./server.js";

const CORS_ORIGIN = "*";
const CORS_METHODS = "GET, POST, DELETE, OPTIONS";
const CORS_ALLOWED_HEADERS = [
  "Content-Type",
  "Authorization",
  "X-Honcho-User-Name",
  "X-Honcho-Workspace-ID",
  "X-Honcho-Assistant-Name",
  "Last-Event-ID",
  "mcp-session-id",
  "mcp-protocol-version",
].join(", ");
const CORS_EXPOSE_HEADERS = "mcp-session-id, mcp-protocol-version";
const MCP_PATHS = new Set(["/", "/mcp"]);

interface BunServer {
  stop: (closeActiveConnections?: boolean) => void;
}

interface BunRuntime {
  env: Record<string, string | undefined>;
  serve: (options: {
    fetch: (request: Request) => Response | Promise<Response>;
    hostname?: string;
    port?: number;
  }) => BunServer;
}

interface ProcessRuntime {
  env: Record<string, string | undefined>;
  on: (event: "SIGINT" | "SIGTERM", listener: () => void | Promise<void>) => void;
  exit: (code?: number) => never;
}

interface SessionState {
  server: ReturnType<typeof createServer>;
  transport: WebStandardStreamableHTTPServerTransport;
}

const bunRuntime = (globalThis as typeof globalThis & { Bun?: BunRuntime }).Bun;
const processRuntime = (globalThis as typeof globalThis & {
  process?: ProcessRuntime;
}).process;

if (!bunRuntime) {
  throw new Error("The Bun MCP server entrypoint must be run with Bun.");
}

const env: Env = {
  HONCHO_API_URL: bunRuntime.env.HONCHO_API_URL ?? processRuntime?.env.HONCHO_API_URL,
};

const sessions = new Map<string, SessionState>();

function jsonResponse(body: unknown, init: ResponseInit = {}): Response {
  return new Response(JSON.stringify(body), {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...corsHeaders(),
      ...init.headers,
    },
  });
}

function corsHeaders(): Record<string, string> {
  return {
    "Access-Control-Allow-Origin": CORS_ORIGIN,
    "Access-Control-Allow-Methods": CORS_METHODS,
    "Access-Control-Allow-Headers": CORS_ALLOWED_HEADERS,
    "Access-Control-Expose-Headers": CORS_EXPOSE_HEADERS,
  };
}

function withCors(response: Response): Response {
  const headers = new Headers(response.headers);
  for (const [key, value] of Object.entries(corsHeaders())) {
    headers.set(key, value);
  }

  return new Response(response.body, {
    status: response.status,
    statusText: response.statusText,
    headers,
  });
}

function mcpErrorResponse(
  status: number,
  message: string,
  id: string | number | null = null,
): Response {
  return jsonResponse(
    {
      jsonrpc: "2.0",
      error: {
        code: -32000,
        message,
      },
      id,
    },
    { status },
  );
}

async function closeSession(sessionId: string): Promise<void> {
  const session = sessions.get(sessionId);
  if (!session) {
    return;
  }

  sessions.delete(sessionId);
  await Promise.allSettled([
    session.server.close(),
    session.transport.close(),
  ]);
}

async function createSession(request: Request): Promise<{
  parsedBody: unknown;
  session: SessionState;
} | Response> {
  let parsedBody: unknown;
  try {
    parsedBody = await request.json();
  } catch {
    return mcpErrorResponse(400, "Parse error: invalid JSON body.");
  }

  if (!isInitializeRequest(parsedBody)) {
    return mcpErrorResponse(
      400,
      "Bad Request: No valid session ID provided.",
    );
  }

  let config;
  try {
    config = parseConfig(request, env);
  } catch (error) {
    const message =
      error instanceof Error ? error.message : "Invalid request";
    return jsonResponse({ error: message }, { status: 401 });
  }

  const honcho = createClient(config);
  const server = createServer({ honcho, config });

  let sessionState: SessionState;
  const transport = new WebStandardStreamableHTTPServerTransport({
    sessionIdGenerator: () => crypto.randomUUID(),
    onsessioninitialized: (sessionId) => {
      sessions.set(sessionId, sessionState);
    },
    onsessionclosed: async (sessionId) => {
      const session = sessions.get(sessionId);
      sessions.delete(sessionId);
      await session?.server.close();
    },
  });

  transport.onerror = (error) => {
    console.error("MCP transport error:", error);
  };

  sessionState = { server, transport };
  await server.connect(transport);

  return {
    parsedBody,
    session: sessionState,
  };
}

async function handleMcpRequest(request: Request): Promise<Response> {
  if (request.method === "OPTIONS") {
    return new Response(null, {
      status: 204,
      headers: corsHeaders(),
    });
  }

  const { pathname } = new URL(request.url);

  if (pathname === "/health") {
    return jsonResponse({ status: "ok" });
  }

  if (!MCP_PATHS.has(pathname)) {
    return new Response("Not Found", { status: 404 });
  }

  if (request.method === "POST") {
    const sessionId = request.headers.get("mcp-session-id");

    if (!sessionId) {
      const createdSession = await createSession(request);
      if (createdSession instanceof Response) {
        return createdSession;
      }

      const response = await createdSession.session.transport.handleRequest(
        request,
        { parsedBody: createdSession.parsedBody },
      );
      return withCors(response);
    }

    const session = sessions.get(sessionId);
    if (!session) {
      return mcpErrorResponse(404, "Session not found.");
    }

    let parsedBody: unknown;
    try {
      parsedBody = await request.json();
    } catch {
      return mcpErrorResponse(400, "Parse error: invalid JSON body.");
    }

    const response = await session.transport.handleRequest(request, {
      parsedBody,
    });
    return withCors(response);
  }

  const sessionId = request.headers.get("mcp-session-id");
  if (!sessionId) {
    return mcpErrorResponse(400, "Bad Request: No valid session ID provided.");
  }

  const session = sessions.get(sessionId);
  if (!session) {
    return mcpErrorResponse(404, "Session not found.");
  }

  const response = await session.transport.handleRequest(request);
  return withCors(response);
}

const host = bunRuntime.env.MCP_HOST || processRuntime?.env.MCP_HOST || "127.0.0.1";
const port = Number.parseInt(
  bunRuntime.env.MCP_PORT || processRuntime?.env.MCP_PORT || "8787",
  10,
);

if (Number.isNaN(port)) {
  throw new Error("MCP_PORT must be a valid integer.");
}

const bunServer = bunRuntime.serve({
  hostname: host,
  port,
  fetch: handleMcpRequest,
});

console.log(`Honcho MCP Bun server listening on http://${host}:${port}`);
if (env.HONCHO_API_URL) {
  console.log(`Routing Honcho API traffic to ${env.HONCHO_API_URL}`);
} else {
  console.log("Routing Honcho API traffic to https://api.honcho.dev");
}

if (processRuntime) {
  for (const signal of ["SIGINT", "SIGTERM"] as const) {
    processRuntime.on(signal, async () => {
      const activeSessionIds = [...sessions.keys()];
      await Promise.allSettled(activeSessionIds.map(closeSession));
      bunServer.stop(true);
      processRuntime.exit(0);
    });
  }
}
