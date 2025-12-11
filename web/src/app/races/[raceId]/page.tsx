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
    popularity?: number;
    odds?: number;
    actual_rank?: number | null;
}

// JRA Venue Code Mapping
const VENUE_MAP: Record<string, string> = {
    "01": "札幌",
    "02": "函館",
    "03": "福島",
    "04": "新潟",
    "05": "東京",
    "06": "中山",
    "07": "中京",
    "08": "京都",
    "09": "阪神",
    "10": "小倉",
};

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
    const [availableVenues, setAvailableVenues] = useState<string[]>([]);
    const [currentDate, setCurrentDate] = useState<string>("");
    const [prevDate, setPrevDate] = useState<string | null>(null);
    const [nextDate, setNextDate] = useState<string | null>(null);
    const [raceRoi, setRaceRoi] = useState<any>(null);  // ROI data for this race
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [mounted, setMounted] = useState(false);
    const [modelId, setModelId] = useState("v12");
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

                        // Use available_venues from API response
                        if (raceData.available_venues) {
                            setAvailableVenues(raceData.available_venues);
                        }

                        // Use date from API response for adjacent dates
                        const raceDate = raceData.race?.date;
                        if (raceDate) {
                            setCurrentDate(raceDate);
                            // Fetch adjacent dates
                            try {
                                const adjRes = await fetch(`http://localhost:8000/api/races/adjacent-dates?date=${raceDate}`);
                                if (adjRes.ok) {
                                    const adjData = await adjRes.json();
                                    setPrevDate(adjData.prev_date || null);
                                    setNextDate(adjData.next_date || null);
                                }
                            } catch (e) {
                                console.warn("Failed to fetch adjacent dates", e);
                            }

                            // Fetch ROI data for this race (same as home page)
                            try {
                                const roiRes = await fetch(`http://localhost:8000/api/daily-roi?date=${raceDate}`);
                                if (roiRes.ok) {
                                    const roiData = await roiRes.json();
                                    if (roiData.by_race && roiData.by_race[raceId]) {
                                        setRaceRoi(roiData.by_race[raceId]);
                                    }
                                }
                            } catch (e) {
                                console.warn("Failed to fetch race ROI", e);
                            }
                        }
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

    if (error || (!loading && !race)) {
        return (
            <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 dark:from-slate-900 dark:to-slate-800 p-8">
                <Card className="max-w-2xl mx-auto border-amber-500 bg-amber-50 dark:bg-amber-950">
                    <CardContent className="pt-6 text-center">
                        <p className="text-4xl mb-4">🏇</p>
                        <p className="font-bold text-xl text-amber-700 dark:text-amber-300 mb-2">
                            レースが見つかりません
                        </p>
                        <p className="text-amber-600 dark:text-amber-400 mb-4">
                            選択された日付にレースがないか、データが存在しません。
                        </p>
                        <p className="text-sm text-slate-500 mb-4">
                            Race ID: {raceId}
                        </p>

                        {/* Guide user to home page to select a race */}
                        <p className="text-sm text-slate-600 mb-4">
                            ホームページからレースを選択してください
                        </p>

                        <div className="flex justify-center gap-2">
                            <Button onClick={() => router.push('/')} variant="default">
                                🏠 ホームへ
                            </Button>
                            <Button onClick={() => router.back()} variant="outline">
                                <ArrowLeft className="mr-2 h-4 w-4" /> 戻る
                            </Button>
                        </div>
                    </CardContent>
                </Card>
            </div>
        );
    }

    return (
        <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 dark:from-slate-900 dark:to-slate-800">
            <div className="container mx-auto px-4 py-8">
                {/* Header with Navigation */}
                <div className="mb-6">
                    {/* Quick Navigation */}
                    <div className="flex flex-wrap items-center gap-2 mb-4 bg-white dark:bg-slate-800 p-3 rounded-lg border">
                        <Button variant="ghost" size="sm" onClick={() => router.back()}>
                            <ArrowLeft className="mr-1 h-4 w-4" /> 戻る
                        </Button>
                        <span className="text-slate-300">|</span>

                        {/* Date Navigation */}
                        <Button
                            variant="ghost"
                            size="sm"
                            onClick={() => prevDate && router.push('/')}
                            disabled={!prevDate}
                            title={prevDate ? `前の開催日: ${prevDate}` : "前の開催日なし"}
                        >
                            ←
                        </Button>

                        <Button variant="outline" size="sm" onClick={() => router.push('/')}>
                            📅 {currentDate || "日付"}
                        </Button>

                        <Button
                            variant="ghost"
                            size="sm"
                            onClick={() => nextDate && router.push('/')}
                            disabled={!nextDate}
                            title={nextDate ? `次の開催日: ${nextDate}` : "次の開催日なし"}
                        >
                            →
                        </Button>
                        <span className="text-slate-300">|</span>

                        {/* Venue Buttons (only show venues available on this date) */}
                        <div className="flex gap-1">
                            {Object.entries(VENUE_MAP)
                                .filter(([code]) => {
                                    // Always show current venue
                                    if (code === race?.venue) return true;
                                    // If no venue data loaded yet, show all
                                    if (availableVenues.length === 0) return true;
                                    // Otherwise filter to available venues
                                    return availableVenues.includes(code);
                                })
                                .map(([code, name]) => (
                                    <Button
                                        key={code}
                                        size="sm"
                                        variant={race?.venue === code ? "default" : "outline"}
                                        className={`px-2 h-7 text-xs ${race?.venue === code ? "bg-indigo-600" : ""}`}
                                        onClick={() => {
                                            // Replace venue code (positions 8-10) and keep race number
                                            const datePrefix = raceId.slice(0, 8); // YYYYMMDD
                                            const venueAndRace = code + raceId.slice(10, 12) + "01"; // venue + kai + day + R01
                                            const newRaceId = datePrefix + venueAndRace;
                                            router.push(`/races/${newRaceId}`);
                                        }}
                                    >
                                        {name}
                                    </Button>
                                ))}
                        </div>

                        <span className="text-slate-300">|</span>

                        {/* Race Number Buttons */}
                        <div className="flex gap-1">
                            {[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12].map((r) => (
                                <Button
                                    key={r}
                                    size="sm"
                                    variant={race?.race_number === r ? "default" : "outline"}
                                    className={`w-7 h-7 p-0 text-xs ${race?.race_number === r ? "bg-indigo-600" : ""}`}
                                    onClick={() => {
                                        const newRaceId = raceId.slice(0, -2) + r.toString().padStart(2, '0');
                                        router.push(`/races/${newRaceId}`);
                                    }}
                                >
                                    {r}
                                </Button>
                            ))}
                        </div>
                    </div>

                    {/* Race Title */}
                    <div className="flex items-center gap-4 mb-3">
                        <Badge className="text-lg px-4 py-2 bg-indigo-600 text-white">
                            {VENUE_MAP[race?.venue || ""] || race?.venue} {race?.race_number}R
                        </Badge>
                        <h1 className="text-2xl md:text-3xl font-bold text-slate-900 dark:text-white">
                            {race?.title || "Race Details"}
                        </h1>
                    </div>

                    {/* Race Info */}
                    <div className="flex flex-wrap gap-3 text-sm">
                        {race?.distance && (
                            <Badge variant="outline" className="font-mono">{race.distance}m</Badge>
                        )}
                        {race?.track_type && (
                            <Badge variant={race.track_type === "芝" ? "default" : "secondary"}
                                className={race.track_type === "芝" ? "bg-green-500" : "bg-amber-500"}>
                                {race.track_type}
                            </Badge>
                        )}
                        {race?.track_condition && (
                            <Badge variant="secondary">{race.track_condition}</Badge>
                        )}
                    </div>

                    {/* ROI Summary (same as home page) */}
                    {raceRoi && (
                        <div className="mt-4 p-3 bg-slate-100 dark:bg-slate-700 rounded-lg">
                            <div className="flex items-center justify-between">
                                <span className="text-sm font-bold text-slate-600 dark:text-slate-300">
                                    📊 推奨買い目収支 ({raceRoi.bet_type})
                                </span>
                                <div className="flex items-center gap-3">
                                    <span className="text-sm">
                                        投資 ¥{raceRoi.cost.toLocaleString()} →
                                        払戻 ¥{raceRoi.return.toLocaleString()}
                                    </span>
                                    {raceRoi.hit ? (
                                        <span className="px-2 py-1 bg-green-500 text-white rounded font-bold text-sm">
                                            ◎ 的中
                                        </span>
                                    ) : (
                                        <span className="px-2 py-1 bg-slate-400 text-white rounded text-sm">
                                            ×
                                        </span>
                                    )}
                                </div>
                            </div>
                        </div>
                    )}
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
                                variant={modelId === "v12" ? "default" : "outline"}
                                onClick={() => setModelId("v12")}
                                className={modelId === "v12" ? "bg-indigo-600 hover:bg-indigo-700" : ""}
                                disabled={isPredicting}
                            >
                                🏆 v12 (High ROI)
                            </Button>
                            <Button
                                size="sm"
                                variant={modelId === "v7" ? "default" : "outline"}
                                onClick={() => setModelId("v7")}
                                className={modelId === "v7" ? "bg-indigo-600 hover:bg-indigo-700" : ""}
                                disabled={isPredicting}
                            >
                                v7 (2025 Best)
                            </Button>
                            <Button
                                size="sm"
                                variant={modelId === "v5" ? "default" : "outline"}
                                onClick={() => setModelId("v5")}
                                className={modelId === "v5" ? "bg-indigo-600 hover:bg-indigo-700" : ""}
                                disabled={isPredicting}
                            >
                                v5
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
                                                <th className="py-2 px-2 text-center">予測</th>
                                                <th className="py-2 px-2 text-center">No.</th>
                                                <th className="py-2 px-2">馬名</th>
                                                <th className="py-2 px-2 text-center">人気</th>
                                                <th className="py-2 px-2 text-center">着順</th>
                                                <th className="py-2 px-2 text-right">確率</th>
                                                <th className="py-2 px-2 text-center">E.V.</th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                            {predictions.map((pred) => (
                                                <tr key={pred.horse_number} className="border-b last:border-0 hover:bg-slate-50 dark:hover:bg-slate-900">
                                                    <td className="py-2 px-2 font-bold text-center">
                                                        {pred.predicted_rank <= 3 ? "🏅" : ""} {pred.predicted_rank}
                                                    </td>
                                                    <td className="py-2 px-2 text-center">
                                                        <span
                                                            className={`inline-flex items-center justify-center w-6 h-6 rounded-full text-xs font-bold border ${frameColors[pred.frame_number] || "bg-gray-200"}`}
                                                        >
                                                            {pred.horse_number}
                                                        </span>
                                                    </td>
                                                    <td className="py-2 px-2 font-medium text-sm">{pred.horse_name}</td>
                                                    <td className="py-2 px-2 text-center">
                                                        {pred.popularity ? (
                                                            <Badge variant={pred.popularity <= 3 ? "default" : "secondary"}
                                                                className={pred.popularity === 1 ? "bg-amber-500" : ""}>
                                                                {pred.popularity}
                                                            </Badge>
                                                        ) : "-"}
                                                    </td>
                                                    <td className="py-2 px-2 text-center font-bold">
                                                        {pred.actual_rank ? (
                                                            <span className={pred.actual_rank <= 3 ? "text-green-600" : ""}>
                                                                {pred.actual_rank}着
                                                            </span>
                                                        ) : "-"}
                                                    </td>
                                                    <td className="py-2 px-2 text-right font-mono text-indigo-600 dark:text-indigo-400 font-bold text-sm">
                                                        {pred.probability ? `${pred.probability}%` : "-"}
                                                    </td>
                                                    <td className="py-2 px-2 text-center font-mono text-sm">
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

                                {/* v12 EV-based Betting Recommendation */}
                                <div className="bg-white dark:bg-slate-950 rounded-lg border border-amber-200 dark:border-amber-800 p-6">
                                    <h3 className="font-bold text-lg text-amber-700 dark:text-amber-400 flex items-center gap-2 mb-4">
                                        🎯 推奨買い目 (v12 EV戦略)
                                    </h3>
                                    {raceRoi ? (
                                        <div className="space-y-4">
                                            {raceRoi.bet_type === "見送り" ? (
                                                <div className="bg-slate-100 dark:bg-slate-900 rounded-lg p-6 text-center border-2 border-dashed border-slate-300 dark:border-slate-700">
                                                    <div className="flex justify-center mb-4">
                                                        <span className="bg-slate-500 text-white px-3 py-1 rounded-full text-sm font-bold">
                                                            見送り (Low EV)
                                                        </span>
                                                    </div>
                                                    <p className="font-bold text-slate-700 dark:text-slate-300 text-lg mb-2">
                                                        期待値が低いため購入対象外
                                                    </p>
                                                    <p className="text-sm text-slate-500 dark:text-slate-400">
                                                        EV &lt; 0.8 のレースは長期的ROIが低いため見送り。
                                                    </p>
                                                </div>
                                            ) : (
                                                <>
                                                    <div className="flex flex-wrap items-center gap-2">
                                                        <Badge className={raceRoi.bet_type === "三連複" ? "bg-blue-500 text-white" : "bg-red-500 text-white"}>
                                                            {raceRoi.bet_type}
                                                        </Badge>
                                                        <span className="font-bold text-slate-700 dark:text-slate-300">
                                                            {raceRoi.bet_type === "三連複" ? "High EV (ROI 142.7%)" : "Mid EV (ROI 116.5%)"}
                                                        </span>
                                                    </div>

                                                    <div className="bg-slate-50 dark:bg-slate-900 rounded-lg p-4">
                                                        {raceRoi.bet_type === "三連複" && predictions.length >= 4 && (
                                                            <div>
                                                                <p className="text-sm text-slate-500 mb-2">1頭軸3頭流し (3点)</p>
                                                                <p className="font-mono font-bold text-lg text-slate-800 dark:text-slate-200">
                                                                    軸: {predictions[0]?.horse_number}番 {predictions[0]?.horse_name}
                                                                </p>
                                                                <p className="font-mono text-slate-600 dark:text-slate-400">
                                                                    → 相手: {predictions.slice(1, 4).map(p => p.horse_number).join(", ")}番
                                                                </p>
                                                            </div>
                                                        )}
                                                        {raceRoi.bet_type === "三連単" && predictions.length >= 4 && (
                                                            <div>
                                                                <p className="text-sm text-slate-500 mb-2">1着固定流し (6点)</p>
                                                                <p className="font-mono font-bold text-lg text-slate-800 dark:text-slate-200">
                                                                    軸: {predictions[0]?.horse_number}番 {predictions[0]?.horse_name}
                                                                </p>
                                                                <p className="font-mono text-slate-600 dark:text-slate-400">
                                                                    → 相手: {predictions.slice(1, 4).map(p => p.horse_number).join(", ")}番
                                                                </p>
                                                            </div>
                                                        )}
                                                    </div>

                                                    {/* Payout Section - from API */}
                                                    <div className="grid grid-cols-3 gap-4 text-center border-t pt-4">
                                                        <div>
                                                            <p className="text-xs text-slate-500">投資額</p>
                                                            <p className="font-bold text-lg">{raceRoi.cost.toLocaleString()}円</p>
                                                        </div>
                                                        <div>
                                                            <p className="text-xs text-slate-500">払戻額</p>
                                                            <p className={`font-bold text-lg ${raceRoi.hit ? "text-green-600" : "text-red-500"}`}>
                                                                {raceRoi.return.toLocaleString()}円
                                                            </p>
                                                        </div>
                                                        <div>
                                                            <p className="text-xs text-slate-500">回収率</p>
                                                            <p className={`font-bold text-lg ${raceRoi.return >= raceRoi.cost ? "text-green-600" : "text-red-500"}`}>
                                                                {raceRoi.cost > 0 ? ((raceRoi.return / raceRoi.cost) * 100).toFixed(1) : 0}%
                                                            </p>
                                                        </div>
                                                    </div>

                                                    <div className={`text-center py-2 rounded ${raceRoi.hit ? "bg-green-100 text-green-700" : "bg-red-100 text-red-700"}`}>
                                                        {raceRoi.hit ? "🎉 的中！" : "❌ 不的中"}
                                                    </div>
                                                </>
                                            )}

                                            <p className="text-xs text-slate-400 mt-4">
                                                ※ v12 EV戦略: High EV≥1.2→三連複3点, Mid EV 0.8-1.2→三連単6点, Low EV→見送り
                                            </p>
                                        </div>
                                    ) : predictions.length >= 5 ? (
                                        <div className="text-center py-4 text-slate-500">
                                            収支データを読み込み中...
                                        </div>
                                    ) : (
                                        <div className="text-center py-4 text-slate-500">
                                            予測データがありません
                                        </div>
                                    )}
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
        </div >
    );
}
