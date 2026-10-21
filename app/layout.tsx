import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "SourceSage",
  description: "Upload a PDF or text file and chat with grounded answers."
};

export default function RootLayout({
  children
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
