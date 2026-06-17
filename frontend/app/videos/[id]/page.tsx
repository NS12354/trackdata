import Link from "next/link";
import { api, anonymizedVideoUrl } from "@/lib/api";
import { Panel } from "@/components/ui";
import VideoDetail from "@/components/VideoDetail";
import type {
  Video, SegmentsResponse, HandPoseResponse, HeadPoseResponse, VideoSummary,
} from "@/lib/types";

export const dynamic = "force-dynamic";

async function safe<T>(p: Promise<T>): Promise<T | null> {
  try {
    return await p;
  } catch {
    return null;
  }
}

export default async function VideoDetailPage({ params }: { params: { id: string } }) {
  const video = await safe<Video>(api.getVideo(params.id));
  if (!video) {
    return (
      <Panel className="p-8 text-center">
        <h1 className="text-lg font-semibold">Video not found</h1>
        <Link href="/videos" className="mt-2 inline-block text-sm text-accent">
          ← Back to videos
        </Link>
      </Panel>
    );
  }

  const [segments, handpose, headpose, summary] = await Promise.all([
    safe<SegmentsResponse>(api.getSegments(params.id)),
    safe<HandPoseResponse>(api.getHandPose(params.id)),
    safe<HeadPoseResponse>(api.getHeadPose(params.id)),
    safe<VideoSummary>(api.getEvents(params.id)),
  ]);

  return (
    <VideoDetail
      video={video}
      segments={segments}
      handpose={handpose}
      headpose={headpose}
      summary={summary}
      videoUrl={anonymizedVideoUrl(params.id, video.anonymized_at)}
    />
  );
}
