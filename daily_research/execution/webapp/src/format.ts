import type { JsonObject, TableRow } from "./types";

export function text(value: unknown, fallback = "-"): string {
  if (value === null || value === undefined || value === "") {
    return fallback;
  }
  return String(value);
}

export function numberValue(value: unknown): number {
  if (typeof value === "number") {
    return Number.isFinite(value) ? value : 0;
  }
  const parsed = Number(String(value ?? "").replace(/,/g, ""));
  return Number.isFinite(parsed) ? parsed : 0;
}

export function percent(value: unknown): string {
  const num = numberValue(value);
  if (Math.abs(num) <= 1) {
    return `${(num * 100).toFixed(2)}%`;
  }
  return `${num.toFixed(2)}%`;
}

export function pick(row: TableRow | JsonObject | undefined, names: string[]): unknown {
  if (!row) {
    return "";
  }
  for (const name of names) {
    const value = row[name];
    if (value !== undefined && value !== null && value !== "") {
      return value;
    }
  }
  return "";
}

export function tableColumns(rows: TableRow[], preferred: string[] = []): string[] {
  const seen = new Set<string>();
  for (const column of preferred) {
    if (rows.some((row) => row[column] !== undefined)) {
      seen.add(column);
    }
  }
  for (const row of rows) {
    for (const key of Object.keys(row)) {
      if (seen.size >= 10) {
        break;
      }
      seen.add(key);
    }
  }
  return Array.from(seen);
}
