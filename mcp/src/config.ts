import { Honcho } from "@honcho-ai/sdk";

export interface HonchoConfig {
  apiKey: string;
  userName: string;
  assistantName: string;
  baseUrl: string;
  workspaceId: string;
}

export interface Env {
  HONCHO_API_URL?: string;
}

const DEFAULT_BASE_URL = "https://api.honcho.dev";

interface HeaderSource {
  get(name: string): string | null;
}

function parseConfigFromHeaders(
  headers: HeaderSource,
  baseUrl: string,
): HonchoConfig {
  const authHeader = headers.get("Authorization");
  const trimmedAuthHeader = authHeader?.trim();
  if (!trimmedAuthHeader?.startsWith("Bearer ")) {
    throw new Error(
      "Missing Authorization header. Provide 'Authorization: Bearer <your-honcho-key>'.",
    );
  }
  const apiKey = trimmedAuthHeader.substring(7).trim();
  if (!apiKey) {
    throw new Error("Authorization header is empty after 'Bearer '.");
  }

  const rawUserName = headers.get("X-Honcho-User-Name");
  const userName = rawUserName?.trim();
  if (!userName) {
    throw new Error(
      "Missing X-Honcho-User-Name header. Provide 'X-Honcho-User-Name: <your-name>'.",
    );
  }

  return {
    apiKey,
    userName,
    assistantName: headers.get("X-Honcho-Assistant-Name")?.trim() || "Assistant",
    baseUrl,
    workspaceId: headers.get("X-Honcho-Workspace-ID")?.trim() || "default",
  };
}

function parseHeaderArgument(rawHeader: string): {
  name: string;
  value: string;
} {
  const delimiter = rawHeader.indexOf(":");
  if (delimiter === -1) {
    throw new Error(
      `Invalid --header value '${rawHeader}'. Expected 'Name:Value'.`,
    );
  }

  const name = rawHeader.slice(0, delimiter).trim();
  const value = rawHeader.slice(delimiter + 1).trim();

  if (!name) {
    throw new Error(`Invalid --header value '${rawHeader}'. Header name is empty.`);
  }

  return { name, value };
}

/**
 * Parse configuration from request headers and Worker env bindings.
 * Throws on missing required fields so callers get clear errors.
 *
 * The Honcho API URL is read from the `HONCHO_API_URL` env var when set,
 * allowing operators to run this Worker alongside a self-hosted Honcho
 * instance (see the "Self-Hosted Honcho" section in README.md). It is
 * intentionally not exposed as a request header: routing public requests
 * to an internal URL would be a latency and security regression.
 */
export function parseConfig(request: Request, env: Env = {}): HonchoConfig {
  return parseConfigFromHeaders(
    request.headers,
    env.HONCHO_API_URL?.trim() || DEFAULT_BASE_URL,
  );
}

export function parseCliConfig(argv: string[], env: Env = {}): HonchoConfig {
  const headers = new Headers();
  let baseUrlArg: string | undefined;

  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];

    if (arg === "--header") {
      const rawHeader = argv[i + 1];
      if (!rawHeader) {
        throw new Error("Missing value after --header.");
      }
      const { name, value } = parseHeaderArgument(rawHeader);
      headers.set(name, value);
      i += 1;
      continue;
    }

    if (arg === "--base-url") {
      const rawBaseUrl = argv[i + 1];
      if (!rawBaseUrl) {
        throw new Error("Missing value after --base-url.");
      }
      baseUrlArg = rawBaseUrl.trim();
      i += 1;
      continue;
    }

    if (arg.startsWith("--")) {
      throw new Error(`Unknown argument '${arg}'.`);
    }

    if (baseUrlArg) {
      throw new Error(
        `Unexpected positional argument '${arg}'. Base URL is already set.`,
      );
    }

    baseUrlArg = arg.trim();
  }

  return parseConfigFromHeaders(
    headers,
    baseUrlArg || env.HONCHO_API_URL?.trim() || DEFAULT_BASE_URL,
  );
}

export function createClient(config: HonchoConfig): Honcho {
  return new Honcho({
    apiKey: config.apiKey,
    baseURL: config.baseUrl,
    workspaceId: config.workspaceId,
  });
}
