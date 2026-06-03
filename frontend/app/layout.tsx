import "./globals.css";
import type { Metadata } from "next";
import { NavLink } from "@/components/ui";

export const metadata: Metadata = {
  title: "Revisent — Operations",
  description: "Waste-services video processing & operational dashboard",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-bg text-text">
        <header className="sticky top-0 z-10 border-b border-border bg-bg/90 backdrop-blur">
          <div className="mx-auto flex max-w-7xl items-center gap-2 px-6 py-3">
            <a href="/" className="mr-4 flex items-center gap-2 font-semibold">
              <span className="inline-block h-2.5 w-2.5 rounded-sm bg-accent" />
              Revisent
            </a>
            <NavLink href="/">Overview</NavLink>
            <NavLink href="/videos">Videos</NavLink>
            <span className="ml-auto text-xs text-muted">operator dashboard</span>
          </div>
        </header>
        <main className="mx-auto max-w-7xl px-6 py-6">{children}</main>
      </body>
    </html>
  );
}
