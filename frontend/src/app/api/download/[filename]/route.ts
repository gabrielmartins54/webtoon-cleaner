import { NextResponse } from 'next/server';

const REMOTE_API_BASE =
  (process.env.API_BASE_URL || process.env.NEXT_PUBLIC_API_BASE_URL || 'http://127.0.0.1:8000').replace(
    /\/+$/,
    '',
  );

type Params = {
  params: Promise<{ filename: string }>;
};

export async function GET(_request: Request, { params }: Params) {
  const { filename } = await params;
  const safeFilename = decodeURIComponent(filename);

  try {
    const upstream = await fetch(`${REMOTE_API_BASE}/download/${encodeURIComponent(safeFilename)}`, {
      method: 'GET',
      cache: 'no-store',
    });

    if (!upstream.ok || !upstream.body) {
      const text = await upstream.text();
      return new NextResponse(text || `Upstream error ${upstream.status}`, {
        status: upstream.status,
        headers: { 'content-type': 'text/plain; charset=utf-8' },
      });
    }

    const headers = new Headers();
    headers.set('content-type', upstream.headers.get('content-type') || 'application/zip');
    headers.set('content-disposition', `attachment; filename="${safeFilename}"`);

    return new Response(upstream.body, {
      status: 200,
      headers,
    });
  } catch (err) {
    const message = err instanceof Error ? err.message : 'Unknown download proxy error';
    return new NextResponse(`Download proxy failed: ${message}`, { status: 502 });
  }
}
