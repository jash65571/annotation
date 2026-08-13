/** One validator gate chip: PASS / REVIEW REQUIRED / FAIL, text + color. */

export function GateBadge({
  label,
  status,
  failCount,
  reviewCount,
}: {
  label: string;
  status?: string;
  failCount?: number;
  reviewCount?: number;
}) {
  let verdict: "PASS" | "REVIEW REQUIRED" | "FAIL" | "NOT RUN";
  if (status !== undefined) {
    verdict =
      status === "PASS"
        ? "PASS"
        : status === "FAIL" || status === "FAILED"
          ? "FAIL"
          : status === "NOT_RUN"
            ? "NOT RUN"
            : "REVIEW REQUIRED";
  } else {
    verdict =
      (failCount ?? 0) > 0 ? "FAIL" : (reviewCount ?? 0) > 0 ? "REVIEW REQUIRED" : "PASS";
  }
  const cls =
    verdict === "PASS"
      ? "pass"
      : verdict === "FAIL"
        ? "fail"
        : verdict === "NOT RUN"
          ? "neutral"
          : "review";
  const counts =
    status === undefined
      ? ` ${failCount ?? 0} fail / ${reviewCount ?? 0} review`
      : "";
  return (
    <span className={`badge ${cls}`} title={`${label}:${counts || ` ${verdict}`}`}>
      {label}: {verdict}
      {counts}
    </span>
  );
}
