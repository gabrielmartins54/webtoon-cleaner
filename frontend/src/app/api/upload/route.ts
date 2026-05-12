import { NextResponse } from 'next/server';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

const REMOTE_API_BASE =
  (process.env.API_BASE_URL || process.env.NEXT_PUBLIC_API_BASE_URL || 'http://127.0.0.1:8000').replace(
    /\/+$/,
    '',
  );

export async function POST(request: Request) {
  try {
    const contentType = request.headers.get('content-type') || 'multipart/form-data';
    const contentLength = request.headers.get('content-length');
    const headers: Record<string, string> = { 'content-type': contentType };
    if (contentLength) headers['content-length'] = contentLength;
    const fetchInit: RequestInit & { duplex: 'half' } = {
      method: 'POST',
      headers,
      body: request.body,
      duplex: 'half',
      cache: 'no-store',
    };

    const upstream = await fetch(`${REMOTE_API_BASE}/upload`, {
      ...fetchInit,
    });

    const text = await upstream.text();
    if (!upstream.ok) {
      return new NextResponse(text || `Upstream error ${upstream.status}`, {
        status: upstream.status,
        headers: { 'content-type': 'text/plain; charset=utf-8' },
      });
    }

    let data: Record<string, unknown>;
    try {
      data = JSON.parse(text) as Record<string, unknown>;
    } catch {
      return new NextResponse('Invalid upstream JSON response', { status: 502 });
    }

    const filename = typeof data.filename === 'string' ? data.filename : null;
    if (filename) {
      data.download_url = `/api/download/${encodeURIComponent(filename)}`;
    }

    return NextResponse.json(data);
  } catch (err) {
    const message = err instanceof Error ? err.message : 'Unknown upload proxy error';
    return new NextResponse(`Upload proxy failed: ${message}`, { status: 502 });
  }
}
