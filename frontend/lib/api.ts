/**
 * Typed client for the GlowMatch API.
 *
 * The base URL was previously re-derived in four places with two different
 * fallback idioms; it lives here now.
 */

export const API_URL =
  process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

// ── Shared shapes ─────────────────────────────────────────────────────────────

export type Product = {
  brand: string;
  product: string;
  shade?: string;
  price_range?: string;
  why?: string;
  url?: string;
};

/** Category keys returned by /analyze, in the order the results page shows them. */
export const CATEGORY_ORDER = [
  "foundation",
  "concealer",
  "blush",
  "bronzer",
  "highlighter",
  "lip",
  "eyeshadow",
  "setting_powder",
  "mascara",
  "brow",
] as const;

export type CategoryKey = (typeof CATEGORY_ORDER)[number];

export const CATEGORY_LABELS: Record<CategoryKey, string> = {
  foundation: "Foundation",
  concealer: "Concealer",
  blush: "Blush",
  bronzer: "Bronzer",
  highlighter: "Highlighter",
  lip: "Lip",
  eyeshadow: "Eyeshadow",
  setting_powder: "Setting powder",
  mascara: "Mascara",
  brow: "Brow",
};

export type Recommendations = Partial<Record<CategoryKey, Product[]>>;

// ── Errors ────────────────────────────────────────────────────────────────────

/** An API failure carrying the user-facing reasons the backend supplied. */
export class ApiError extends Error {
  readonly status: number;
  /** The backend returns `detail` as either a string or a list of strings. */
  readonly reasons: string[];

  constructor(status: number, reasons: string[]) {
    super(reasons[0] ?? `Request failed (${status})`);
    this.name = "ApiError";
    this.status = status;
    this.reasons = reasons;
  }
}

async function toApiError(response: Response): Promise<ApiError> {
  let reasons: string[] = [];
  try {
    const body = await response.json();
    const detail = body?.detail;
    if (Array.isArray(detail)) reasons = detail.map(String);
    else if (typeof detail === "string") reasons = [detail];
  } catch {
    // Non-JSON error body — fall through to the status-based default.
  }

  if (reasons.length === 0) {
    reasons = [
      response.status === 429
        ? "Too many requests just now. Give it a moment and try again."
        : response.status === 503
          ? "GlowMatch has hit its daily usage limit. Try again tomorrow."
          : "Something went wrong on our end. Please try again.",
    ];
  }
  return new ApiError(response.status, reasons);
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_URL}${path}`, init);
  } catch {
    // Network-level failure: no response at all.
    throw new ApiError(0, [
      "Couldn't reach the server. Check your connection and try again.",
    ]);
  }
  if (!response.ok) throw await toApiError(response);
  return response.json() as Promise<T>;
}

function postJson<T>(path: string, body: unknown): Promise<T> {
  return request<T>(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

// ── Wake ──────────────────────────────────────────────────────────────────────

/**
 * Fire-and-forget ping so a sleeping Render free-tier instance (~30s cold
 * start) is booting while the user is still choosing a photo, not after.
 */
export function wakeServer(): void {
  fetch(`${API_URL}/health`).catch(() => {
    // Best effort — real requests surface their own errors.
  });
}

// ── Analyze ───────────────────────────────────────────────────────────────────

export type MatchedShade = {
  shade_name: string;
  hex: string;
  description: string;
  recommendation?: string;
  match_score?: number;
  /** CIEDE2000 to the measured skin tone. ~2.3 is the just-noticeable difference. */
  delta_e?: number;
  /** Plain-language reading of delta_e, from the backend. */
  closeness?: string;
};

export type AnalyzeResult = {
  pixel_count: number;
  monk_scale: string;
  undertone: string;
  avg_hex: string;
  matched_shades: MatchedShade[];
  recommendations: Recommendations;
};

export function analyze(file: File, budget: string): Promise<AnalyzeResult> {
  const form = new FormData();
  form.append("file", file);
  form.append("budget", budget);
  return request<AnalyzeResult>("/analyze", { method: "POST", body: form });
}

// ── Product images ────────────────────────────────────────────────────────────

/** Matches the backend's cache key so lookups line up on both sides. */
export function imageKey(brand: string, product: string): string {
  return `${brand.trim().toLowerCase()}|${product.trim().toLowerCase()}`;
}

export function fetchProductImages(
  products: { brand: string; product: string }[],
): Promise<{ images: Record<string, string | null> }> {
  return postJson("/product-images", { products });
}

// ── Skincare quiz ─────────────────────────────────────────────────────────────

export type QuizOption = { value: string; label: string; hint: string };

export type QuizQuestion = {
  id: string;
  prompt: string;
  multi: boolean;
  help_text: string;
  options: QuizOption[];
};

export type QuizAnswers = Record<string, string | string[]>;

export type RoutineProduct = {
  step: string;
  brand: string;
  product: string;
  price_range: string;
  key_ingredient: string;
  why: string;
  when: string;
  url: string;
};

export type QuizResponse = {
  profile: {
    top_tags: string[];
    tag_scores: Record<string, number>;
    sensitive: boolean;
    beginner: boolean;
    rationale: Record<string, string[]>;
  };
  routine_steps: string[];
  routine: RoutineProduct[];
};

export function fetchQuizQuestions(): Promise<{ questions: QuizQuestion[] }> {
  return request("/skincare-quiz/questions");
}

export function submitQuiz(
  answers: QuizAnswers,
  budget: string,
): Promise<QuizResponse> {
  return postJson("/skincare-quiz", { answers, budget });
}

// ── Lip palette ───────────────────────────────────────────────────────────────

export type LipFamily = {
  name: string;
  hex: string;
  note: string;
  undertone_affinity: string;
  in_depth_range: boolean;
  why: string;
};

/** Lip colour families ranked for a measured tone. Colours, not products. */
export function fetchLipPalette(
  undertone: string,
  mstLevel: number,
): Promise<{ undertone: string; mst_level: number; families: LipFamily[] }> {
  const params = new URLSearchParams({
    undertone,
    mst_level: String(mstLevel),
  });
  return request(`/lip-palette?${params}`);
}

// ── Product lookup (inventory search) ─────────────────────────────────────────

export type ProductMatch = {
  brand: string;
  product: string;
  category: string | null;
  shade: string | null;
  hex: string | null;
  /** Typical retail price in cents. An estimate, not live pricing. */
  price_cents: number | null;
};

export function lookupProducts(query: string): Promise<{ results: ProductMatch[] }> {
  return postJson("/product-lookup", { query });
}

export type ProductShade = {
  name: string;
  hex: string | null;
  undertone: "warm" | "cool" | "neutral" | null;
};

/** The shade range for one product. Separate from lookupProducts because a
 *  shade name is only meaningful once the product is known. */
export function lookupShades(
  brand: string,
  product: string,
): Promise<{ shades: ProductShade[] }> {
  return postJson("/product-shades", { brand, product });
}
