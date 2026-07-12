export default function EmailLink() {
  const email = 'hasson827624@gmail.com';

  return (
    <div className="contact-email-container">
      <a
        href={`mailto:${email}`}
        className="contact-email-link"
        aria-label={`Email ${email}`}
      >
        <span className="contact-email-prefix">{email}</span>
      </a>
    </div>
  );
}
