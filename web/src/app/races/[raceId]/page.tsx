"use client";

import { useState, useEffect } from "react";
import { useParams, useRouter } from "next/navigation";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { ArrowLeft, Trophy, Users } from "lucide-react";

interface RaceInfo {
    race_id: string;
    venue: string;
    race_number: number;
    title: string;
    distance: number;
    track_type: string;
    track_condition: string;
}

interface HorseEntry {
    horse_number: number;
    frame_number: number;
    horse_name: string;
    sex: string;
    age: number;
    jockey_name: string;
    trainer_name: string;
    weight: number;
    weight_diff: number;
    odds: number;
    popularity: number;
    horse_id: string;
}

interface Prediction {
    horse_number: number;
    frame_number: number;
    horse_name: string;
    score: number;
    probability?: number;
    expected_value: number;
    predicted_rank: number;
}

// Frame colors (枠番の色)
const frameColors: Record<number, string> = {
    1: "bg-white border-gray-300 text-gray-900",
    2: "bg-black text-white",
    3: "bg-red-500 text-white",
    4: "bg-blue-500 text-white",
    5: "bg-yellow-400 text-gray-900",
    6: "bg-green-500 text-white",
    7: "bg-orange-500 text-white",
    8: "bg-pink-400 text-white",
};

export default function RaceDetailPage() {
    const params = useParams();
    const router = useRouter();
    const raceId = params.raceId as string;

    const [race, setRace] = useState<RaceInfo | null>(null);
    const [entries, setEntries] = useState<HorseEntry[]>([]);
    const [predictions, setPredictions] = useState<Prediction[]>([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [mounted, setMounted] = useState(false);
    const [modelId, setModelId] = useState("v4_2025");
    const [isPredicting, setIsPredicting] = useState(false);

    useEffect(() => {
        setMounted(true);
    }, []);

    useEffect(() => {
        const fetchAllData = async () => {
            if (!mounted) return;
            // Initial load
            if (!race) setLoading(true);

            // Prediction load specifically
            setIsPredicting(true);
            setError(null);

            try {
                if (!raceId) return;

                // Fetch Race Detail (only if not loaded)
                if (!race) {
                    const raceRes = await fetch(`http://localhost:8000/api/races/${raceId}`);
                    if (!raceRes.ok) throw new Error("Failed to fetch race details");
                    const raceData = await raceRes.json();

                    if (raceData.message && !raceData.race) {
                        setRace(null);
                        setEntries([]);
                        setError(raceData.message);
                        setLoading(false);
                        setIsPredicting(false);
                        return;
                    } else {
                        setRace(raceData.race);
                        setEntries(raceData.entries || []);
                    }
                }

                // Fetch Predictions
                try {
                    const predRes = await fetch(`http://localhost:8000/api/races/${raceId}/predictions?model_id=${modelId}`);
                    if (predRes.ok) {
                        const predData = await predRes.json();
                        setPredictions(predData.predictions || []);
                    } else {
                        console.warn("Predictions not available");
                        setPredictions([]);
                    }
                } catch (e) {
                    console.warn("Failed to fetch predictions", e);
                }

            } catch (err) {
                console.error("Catch error:", err);
                setError(err instanceof Error ? err.message : "Unknown error");
            } finally {
                setLoading(false);
                setIsPredicting(false);
            }
        };

        if (raceId && mounted) {
            fetchAllData();
        }
    }, [raceId, mounted, modelId]);

    // Helper to calculate heatmap color for EV
    const maxEV = Math.max(...predictions.map(p => p.expected_value || 0), 0.1); // Avoid 0 division

    const getEvStyle = (ev: number) => {
        if (!ev) return {};
        // Normalize 0 to 1 based on max EV
        const intensity = Math.min(ev / maxEV, 1);

        // Use an amber/orange scale for heat
        // Light: 255, 247, 237 (amber-50) -> Dark: 245, 158, 11 (amber-500)
        // Simplest way is to set opacity of a background color
        return {
            backgroundColor: `rgba(245, 158, 11, ${intensity * 0.4})`, // Max 40% opacity
            fontWeight: intensity > 0.7 ? 'bold' : 'normal'
        };
    };

    if (!mounted) return null;

    if (loading && !race) {
        return (
            <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 dark:from-slate-900 dark:to-slate-800 flex items-center justify-center">
                <div className="text-center">
                    <div className="inline-block animate-spin rounded-full h-12 w-12 border-b-2 border-indigo-600"></div>
                    <p className="mt-4 text-slate-600 dark:text-slate-400">Loading race details...</p>
                </div>
            </div>
        );
    }

    if (error) {
        return (
            <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 dark:from-slate-900 dark:to-slate-800 p-8">
                <Card className="max-w-2xl mx-auto border-red-500 bg-red-50 dark:bg-red-950">
                    <CardContent className="pt-6">
                        <p className="font-bold text-lg text-red-700 dark:text-red-300">Something went wrong:</p>
                        <pre className="mt-2 p-4 bg-black/10 rounded text-sm text-red-600 dark:text-red-400 overflow-auto">
                            {error}
                        </pre>
                        <Button onClick={() => window.location.reload()} className="mt-4 mr-2">
                            Retry
                        </Button>
                        <Button onClick={() => router.back()} variant="outline" className="mt-4">
                            <ArrowLeft className="mr-2 h-4 w-4" /> Go Back
                        </Button>
                    </CardContent>
                </Card>
            </div>
        );
    }

    return (
        <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 dark:from-slate-900 dark:to-slate-800">
            <div className="container mx-auto px-4 py-8">
                {/* Header */}
                <div className="mb-6">
                    <Button variant="ghost" onClick={() => router.back()} className="mb-4">
                        <ArrowLeft className="mr-2 h-4 w-4" /> Back to List
                    </Button>

                    <div className="flex items-center gap-4 mb-2">
                        <Badge variant="outline" className="text-lg px-3 py-1">
                            {race?.venue} R{race?.race_number}
                        </Badge>
                        <h1 className="text-3xl font-bold text-slate-900 dark:text-white">
                            {race?.title || "Race Details"}
                        </h1>
                    </div>

                    <div className="flex gap-4 text-slate-600 dark:text-slate-400">
                        {race?.distance && <span>{race.distance}m</span>}
                        {race?.track_type && <span>{race.track_type}</span>}
                        {race?.track_condition && <Badge variant="secondary">{race.track_condition}</Badge>}
                    </div>
                </div>

                {/* AI Predictions Section */}
                <Card className="mb-8 border-indigo-200 dark:border-indigo-800 bg-indigo-50/50 dark:bg-indigo-950/20">
                    <CardHeader className="flex flex-row items-center justify-between pb-2">
                        <div>
                            <CardTitle className="flex items-center gap-2 text-indigo-700 dark:text-indigo-300">
                                <Trophy className="h-5 w-5 text-amber-500" />
                                AI Predictions
                            </CardTitle>
                            <CardDescription>
                                Model: <span className="font-mono font-bold text-indigo-600">{modelId}</span>
                            </CardDescription>
                        </div>
                        <div className="flex gap-2">
                            <Button
                                size="sm"
                                variant={modelId === "v5" ? "default" : "outline"}
                                onClick={() => setModelId("v5")}
                                className={modelId === "v5" ? "bg-indigo-600 hover:bg-indigo-700" : ""}
                                disabled={isPredicting}
                            >
                                JRA Only (v5)
                            </Button>
                            <Button
                                size="sm"
                                variant={modelId === "v4_2025" ? "default" : "outline"}
                                onClick={() => setModelId("v4_2025")}
                                className={modelId === "v4_2025" ? "bg-indigo-600 hover:bg-indigo-700" : ""}
                                disabled={isPredicting}
                            >
                                JRA + 地方 (v4_2025)
                            </Button>
                        </div>
                    </CardHeader>
                    <CardContent>
                        {isPredicting ? (
                            <div className="flex flex-col items-center justify-center py-12">
                                <div className="animate-spin rounded-full h-10 w-10 border-b-2 border-indigo-600 mb-4"></div>
                                <p className="text-slate-500">Calculating predictions with {modelId}...</p>
                            </div>
                        ) : predictions.length > 0 ? (
                            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                                {/* Prediction Table */}
                                <div className="overflow-x-auto bg-white dark:bg-slate-950 rounded-lg border shadow-sm">
                                    <table className="w-full text-sm">
                                        <thead>
                                            <tr className="border-b text-left text-slate-500 bg-slate-50 dark:bg-slate-900">
                                                <th className="py-2 px-3 text-center">Rank</th>
                                                <th className="py-2 px-3 text-center">No.</th>
                                                <th className="py-2 px-3">Horse</th>
                                                <th className="py-2 px-3 text-right">Probability</th>
                                                <th className="py-2 px-3 text-right">Score</th>
                                                <th className="py-2 px-3 text-center">E.V.</th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                            {predictions.map((pred) => (
                                                <tr key={pred.horse_number} className="border-b last:border-0 hover:bg-slate-50 dark:hover:bg-slate-900">
                                                    <td className="py-2 px-3 font-bold text-center">
                                                        {pred.predicted_rank <= 3 ? "🏅" : ""} {pred.predicted_rank}
                                                    </td>
                                                    <td className="py-2 px-3 text-center">
                                                        <span
                                                            className={`inline-flex items-center justify-center w-6 h-6 rounded-full text-xs font-bold border ${frameColors[pred.frame_number] || "bg-gray-200"}`}
                                                        >
                                                            {pred.horse_number}
                                                        </span>
                                                    </td>
                                                    <td className="py-2 px-3 font-medium">{pred.horse_name}</td>
                                                    <td className="py-2 px-3 text-right font-mono text-indigo-600 dark:text-indigo-400 font-bold">
                                                        {pred.probability ? `${pred.probability}%` : "-"}
                                                    </td>
                                                    <td className="py-2 px-3 text-right text-slate-500">
                                                        {pred.score.toFixed(2)}
                                                    </td>
                                                    <td className="py-2 px-3 text-center font-mono">
                                                        {pred.expected_value !== undefined ? (
                                                            <div
                                                                className="py-1 px-2 rounded"
                                                                style={getEvStyle(pred.expected_value)}
                                                            >
                                                                {pred.expected_value.toFixed(2)}
                                                            </div>
                                                        ) : "-"}
                                                    </td>
                                                </tr>
                                            ))}
                                        </tbody>
                                    </table>
                                </div>

                                <div className="flex flex-col items-center justify-center bg-white dark:bg-slate-950 rounded-lg border border-dashed border-slate-300 dark:border-slate-700 p-6 text-center">
                                    <Trophy className="h-12 w-12 text-slate-300 mb-2" />
                                    <h3 className="font-semibold text-slate-700 dark:text-slate-300">Analysis Visualization</h3>
                                    <p className="text-slate-500 text-sm mt-1">
                                        Comparing {modelId} predictions.<br />
                                        Radar charts & time-series coming soon.
                                    </p>
                                </div>
                            </div>
                        ) : (
                            <div className="text-center py-8 text-slate-500">
                                No predictions available for this model yet.
                            </div>
                        )}
                    </CardContent>
                </Card>

                {/* Race Card (出馬表) */}
                <Card>
                    <CardHeader>
                        <CardTitle className="flex items-center gap-2">
                            <Users className="h-5 w-5" />
                            出馬表 (Race Card)
                        </CardTitle>
                        <CardDescription>{entries.length} horses entered</CardDescription>
                    </CardHeader>
                    <CardContent>
                        <div className="overflow-x-auto">
                            <table className="w-full text-sm">
                                <thead>
                                    <tr className="border-b text-left text-slate-500 dark:text-slate-400">
                                        <th className="py-3 px-2 w-12">枠</th>
                                        <th className="py-3 px-2 w-12">番</th>
                                        <th className="py-3 px-2">馬名</th>
                                        <th className="py-3 px-2">性齢</th>
                                        <th className="py-3 px-2">騎手</th>
                                        <th className="py-3 px-2 text-right">斤量</th>
                                        <th className="py-3 px-2 text-right">オッズ</th>
                                        <th className="py-3 px-2 text-right">人気</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {entries.map((horse) => (
                                        <tr
                                            key={horse.horse_number}
                                            className="border-b hover:bg-slate-50 dark:hover:bg-slate-800 transition-colors"
                                        >
                                            <td className="py-3 px-2">
                                                <span
                                                    className={`inline-flex items-center justify-center w-8 h-8 rounded-full text-sm font-bold border ${frameColors[horse.frame_number] || "bg-gray-200"
                                                        }`}
                                                >
                                                    {horse.frame_number}
                                                </span>
                                            </td>
                                            <td className="py-3 px-2 font-bold text-lg">{horse.horse_number}</td>
                                            <td className="py-3 px-2 font-medium">{horse.horse_name || "-"}</td>
                                            <td className="py-3 px-2 text-slate-600 dark:text-slate-400">
                                                {horse.sex}{horse.age}
                                            </td>
                                            <td className="py-3 px-2">{horse.jockey_name || "-"}</td>
                                            <td className="py-3 px-2 text-right">
                                                {horse.weight || "-"}
                                                {horse.weight_diff !== undefined && horse.weight_diff !== 0 && (
                                                    <span className={horse.weight_diff > 0 ? "text-red-500" : "text-blue-500"}>
                                                        ({horse.weight_diff > 0 ? "+" : ""}{horse.weight_diff})
                                                    </span>
                                                )}
                                            </td>
                                            <td className="py-3 px-2 text-right font-mono">
                                                {horse.odds ? horse.odds.toFixed(1) : "-"}
                                            </td>
                                            <td className="py-3 px-2 text-right">
                                                {horse.popularity && horse.popularity <= 3 ? (
                                                    <Badge
                                                        variant={horse.popularity === 1 ? "default" : "secondary"}
                                                        className={horse.popularity === 1 ? "bg-amber-500" : ""}
                                                    >
                                                        {horse.popularity}
                                                    </Badge>
                                                ) : (
                                                    horse.popularity || "-"
                                                )}
                                            </td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    </CardContent>
                </Card>
            </div>
        </div>
    );
}
