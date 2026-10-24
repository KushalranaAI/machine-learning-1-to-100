import Link from "next/link";

export default function NotFound() {
  return (<><h1>Not found</h1><p>That page doesn&apos;t exist. <Link href="/">Back to the roadmap</Link>.</p></>);
}
