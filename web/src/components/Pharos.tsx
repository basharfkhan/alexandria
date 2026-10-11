/** The Pharos of Alexandria: three tiers (square, octagonal, cylindrical) and a lit beacon.
 *  The lighthouse rather than a scroll or an eye, because it is what the city was known for and
 *  it says "this helps you find something" without a word of explanation. */
export function Pharos({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 24 24" fill="none" aria-hidden="true" className={className}>
      <path d="M12 1.4c1.25 1.4 1.87 2.43 1.87 3.27a1.87 1.87 0 1 1-3.74 0c0-.84.62-1.87 1.87-3.27Z" fill="var(--gold)" />
      <path d="m4.8 5.4-2.3-1.2M19.2 5.4l2.3-1.2" stroke="var(--gold)" strokeWidth="1.3" strokeLinecap="round" />
      <path
        d="M9.9 6.6h4.2v3H9.9zM8.7 9.6h6.6l.9 5.1H7.8zM7.2 14.7h9.6l1.2 7.3H6z"
        stroke="currentColor"
        strokeWidth="1.3"
        strokeLinejoin="round"
      />
    </svg>
  );
}
