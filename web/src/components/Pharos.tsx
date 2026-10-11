/** The Pharos of Alexandria: three tiers (square, octagonal, cylindrical) and a lit beacon.
 *  The lighthouse rather than a scroll or an eye, because it is what the city was known for and
 *  it says "this helps you find something" without a word of explanation.
 *
 *  Drawn as solid steps rather than outlines: at 24px an outlined, evenly tapered tower reads as
 *  a plain triangle. The silhouette has to do the work, so each tier is markedly narrower than the
 *  one below it (15 units, then 9, then 4) and a ledge juts out between them. */
export function Pharos({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true" className={className}>
      <g fill="currentColor">
        <path d="M4.6 22l1.8-6.6h11.2L19.4 22z" />
        <path d="M5.9 14.1h12.2v1.3H5.9z" />
        <path d="M7.5 14.1l1.4-5h6.2l1.4 5z" />
        <path d="M8.4 7.9h7.2v1.2H8.4z" />
        <path d="M10 5h4v2.9h-4z" />
      </g>
      <g fill="var(--gold)">
        <path d="M12 .9c1.5 1.7 2.2 2.8 2.2 3.7a2.2 2.2 0 0 1-4.4 0c0-.9.7-2 2.2-3.7z" />
      </g>
      {/* Light cast sideways from the lantern. Angled rays read as antennae at this size. */}
      <path
        d="M5.9 6.4h3.2M14.9 6.4h3.2"
        stroke="var(--gold)"
        strokeWidth="1.5"
        strokeLinecap="round"
      />
    </svg>
  );
}
