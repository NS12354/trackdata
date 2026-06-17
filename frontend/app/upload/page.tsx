import UploadForm from "@/components/UploadForm";

export const metadata = { title: "Upload — Revisent" };

export default function UploadPage() {
  return (
    <div className="space-y-4">
      <div>
        <h1 className="text-xl font-semibold">Upload video</h1>
        <p className="text-sm text-muted">
          Head- or chest-mounted egocentric footage. Processing runs locally and free.
        </p>
      </div>
      <UploadForm />
    </div>
  );
}
