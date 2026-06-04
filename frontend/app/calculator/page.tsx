import CostCalculator from "@/components/CostCalculator";

export const metadata = { title: "GPU Cost Calculator — Revisent" };

export default function CalculatorPage() {
  return (
    <div className="space-y-4">
      <div>
        <h1 className="text-xl font-semibold">GPU cost calculator</h1>
        <p className="text-sm text-muted">
          Estimate the cost to process footage on a GPU. All figures are estimates —
          benchmark on a real GPU to firm up throughput.
        </p>
      </div>
      <CostCalculator />
    </div>
  );
}
