"use client";

import { useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle, CardFooter } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Trash2, Plus, Play, ChevronRight, TrendingUp, DollarSign, Target } from "lucide-react";
import { cn } from "@/lib/utils";

// --- Types ---

// --- Types ---

interface Strategy {
    id: string;
    type: string;
    name: string;
    formation: number[][]; // [ [1], [2,3], [2,3,4] ]
}

// Hierarchical Result Type
interface DailyStat {
    date: string;
    bet: number;
    return: number;
    hit: number;
    race_count: number;
    races: {
        race_id: string;
        title: string;
        bet: number;
        return: number;
        hit: boolean;
    }[];
}

interface StrategyResult {
    summary: {
        roi: number;
        total_profit: number;
        total_bet: number;
        race_count: number;
        hit_rate: number;
    };
    daily: DailyStat[];
}

// Map: ModelKey -> { StrategyName -> StrategyResult }
type SimulationResponse = Record<string, Record<string, StrategyResult>>;

// --- Constants ---

const TICKET_TYPES = [
    { value: "SanRenTan", label: "３連単" },
    { value: "SanRenPuku", label: "３連複" },
    { value: "Umaren", label: "馬連" },
    { value: "Wide", label: "ワイド" },
    { value: "Tansho", label: "単勝" },
];

// --- Sub-components for Drill-down ---

const MonthlyBreakdown = ({ dailyStats }: { dailyStats: DailyStat[] }) => {
    // Group daily stats by month
    const monthlyGroups = dailyStats.reduce((acc, stat) => {
        const month = stat.date.substring(0, 7); // YYYY-MM
        if (!acc[month]) acc[month] = [];
        acc[month].push(stat);
        return acc;
    }, {} as Record<string, DailyStat[]>);

    const sortedMonths = Object.keys(monthlyGroups).sort();
    const [expandedMonth, setExpandedMonth] = useState<string | null>(null);
    const [expandedDate, setExpandedDate] = useState<string | null>(null);

    return (
        <div className="space-y-2 mt-4">
            {sortedMonths.map(month => {
                const days = monthlyGroups[month];
                const totalBet = days.reduce((sum, d) => sum + d.bet, 0);
                const totalReturn = days.reduce((sum, d) => sum + d.return, 0);
                const roi = totalBet > 0 ? (totalReturn / totalBet) * 100 : 0;
                const profit = totalReturn - totalBet;

                const isExpanded = expandedMonth === month;

                return (
                    <div key={month} className="border rounded-md bg-white dark:bg-slate-950 overflow-hidden">
                        <div
                            className="flex items-center justify-between p-3 cursor-pointer hover:bg-slate-50 dark:hover:bg-slate-900 transition-colors"
                            onClick={() => setExpandedMonth(isExpanded ? null : month)}
                        >
                            <div className="flex items-center gap-2">
                                <ChevronRight className={`h-4 w-4 transition-transform ${isExpanded ? "rotate-90" : ""}`} />
                                <span className="font-bold font-mono">{month}</span>
                                <span className="text-xs text-slate-500">({days.length} days)</span>
                            </div>
                            <div className="flex gap-4 text-sm font-mono">
                                <span className={roi >= 100 ? "text-green-600 font-bold" : "text-red-500"}>
                                    ROI: {roi.toFixed(1)}%
                                </span>
                                <span className={profit >= 0 ? "text-green-600" : "text-red-500"}>
                                    {profit > 0 ? "+" : ""}{profit.toLocaleString()}
                                </span>
                            </div>
                        </div>

                        {/* Daily List */}
                        {isExpanded && (
                            <div className="border-t bg-slate-50 dark:bg-slate-900/50 p-2 space-y-1">
                                {days.map(day => {
                                    const dayIsExpanded = expandedDate === day.date;
                                    const dayProfit = day.return - day.bet;
                                    const dayRoi = day.bet > 0 ? (day.return / day.bet) * 100 : 0;

                                    return (
                                        <div key={day.date} className="pl-4">
                                            <div
                                                className="flex items-center justify-between py-1 px-2 rounded hover:bg-white dark:hover:bg-slate-800 cursor-pointer text-sm"
                                                onClick={(e) => {
                                                    e.stopPropagation();
                                                    setExpandedDate(dayIsExpanded ? null : day.date);
                                                }}
                                            >
                                                <div className="flex items-center gap-2">
                                                    <span className={`w-2 h-2 rounded-full ${dayProfit >= 0 ? "bg-green-500" : "bg-red-500"}`}></span>
                                                    <span className="font-mono">{day.date}</span>
                                                </div>
                                                <div className="flex gap-4 font-mono text-xs">
                                                    <span>{day.race_count}R</span>
                                                    <span className={dayRoi >= 100 ? "text-green-600" : "text-red-500"}>{dayRoi.toFixed(0)}%</span>
                                                    <span>{dayProfit.toLocaleString()}</span>
                                                </div>
                                            </div>

                                            {/* Race Detail */}
                                            {dayIsExpanded && (
                                                <div className="mt-1 ml-6 space-y-1 border-l-2 border-slate-200 dark:border-slate-700 pl-2">
                                                    {day.races.map(race => (
                                                        <a
                                                            key={race.race_id}
                                                            href={`/races/${race.race_id}`}
                                                            target="_blank"
                                                            className="flex items-center justify-between text-xs py-1 px-2 hover:bg-indigo-50 dark:hover:bg-indigo-900/30 rounded group block"
                                                        >
                                                            <div className="flex items-center gap-2">
                                                                <span className="text-slate-500">{race.title}</span>
                                                                {race.hit && <Badge variant="default" className="h-4 px-1 text-[10px] bg-green-500">Hit</Badge>}
                                                            </div>
                                                            <div className="font-mono text-slate-500 group-hover:text-indigo-600">
                                                                {race.return - race.bet > 0 ? "+" : ""}{(race.return - race.bet).toLocaleString()}
                                                            </div>
                                                        </a>
                                                    ))}
                                                </div>
                                            )}
                                        </div>
                                    );
                                })}
                            </div>
                        )}
                    </div>
                );
            })}
        </div>
    );
};

export default function SimulationPage() {
    // Config State
    const [selectedModels, setSelectedModels] = useState<string[]>(["v5_2025"]);
    const [selectedYear, setSelectedYear] = useState<number>(2025);
    const [surfaceFilter, setSurfaceFilter] = useState<"All" | "芝" | "ダート">("All");

    // Strategies State
    const [strategies, setStrategies] = useState<Strategy[]>([
        {
            id: "1",
            type: "SanRenTan",
            name: "本命フォーメーション",
            formation: [[1], [2, 3], [2, 3, 4, 5]]
        }
    ]);

    // Result State
    const [result, setResult] = useState<SimulationResponse | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    // --- Actions ---

    const toggleModel = (value: string) => {
        if (selectedModels.includes(value)) {
            setSelectedModels(selectedModels.filter(m => m !== value));
        } else {
            setSelectedModels([...selectedModels, value]);
        }
    };

    const addStrategy = () => {
        const newId = Math.random().toString(36).substr(2, 9);
        setStrategies([...strategies, {
            id: newId,
            type: "SanRenTan",
            name: `戦略 ${strategies.length + 1}`,
            formation: [[1], [2], [3]]
        }]);
    };

    const removeStrategy = (id: string) => {
        setStrategies(strategies.filter(s => s.id !== id));
    };

    const updateStrategy = (id: string, updates: Partial<Strategy>) => {
        setStrategies(strategies.map(s => s.id === id ? { ...s, ...updates } : s));
    };

    const toggleRank = (strategyId: string, rowIndex: number, rank: number) => {
        const strategy = strategies.find(s => s.id === strategyId);
        if (!strategy) return;
        const newFormation = [...strategy.formation];
        while (newFormation.length <= rowIndex) newFormation.push([]);
        const row = newFormation[rowIndex];
        if (row.includes(rank)) {
            newFormation[rowIndex] = row.filter(r => r !== rank).sort((a, b) => a - b);
        } else {
            newFormation[rowIndex] = [...row, rank].sort((a, b) => a - b);
        }
        updateStrategy(strategyId, { formation: newFormation });
    };

    const runSimulation = async () => {
        if (selectedModels.length === 0) {
            setError("モデルを少なくとも1つ選択してください");
            return;
        }
        setLoading(true);
        setError(null);
        setResult(null);

        try {
            const filters = surfaceFilter !== "All" ? { surface: surfaceFilter } : {};

            const payload = {
                model_keys: selectedModels,
                year: selectedYear,
                filters: filters,
                strategies: strategies.map(s => ({
                    type: s.type,
                    name: s.name,
                    formation: s.formation
                }))
            };

            const res = await fetch("http://localhost:8000/api/simulation/run", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            });

            if (!res.ok) {
                const errData = await res.json();
                throw new Error(errData.detail || "Simulation failed");
            }

            const data = await res.json();
            setResult(data);

        } catch (e) {
            setError(e instanceof Error ? e.message : "Unknown error");
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="min-h-screen bg-slate-50 dark:bg-slate-900 font-sans text-slate-900 dark:text-slate-100 p-6">
            <div className="container mx-auto max-w-6xl">
                {/* Header */}
                <div className="mb-8 flex items-center justify-between">
                    <div>
                        <h1 className="text-3xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-indigo-600 to-violet-600 dark:from-indigo-400 dark:to-violet-400">
                            Betting Simulator
                        </h1>
                        <p className="text-slate-500 dark:text-slate-400 mt-1">
                            予測モデルと戦略の収支シミュレーション
                        </p>
                    </div>
                </div>

                {/* Main Content Grid */}
                <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">

                    {/* LEFT COLUMN: Config & Strategies */}
                    <div className="lg:col-span-5 flex flex-col gap-6">

                        {/* Config Panel */}
                        <Card className="border-indigo-100 dark:border-indigo-900/50 shadow-sm">
                            <CardHeader className="pb-3">
                                <CardTitle className="text-lg flex items-center gap-2">
                                    <Target className="h-5 w-5 text-indigo-500" />
                                    設定 (Settings)
                                </CardTitle>
                            </CardHeader>
                            <CardContent className="space-y-4">
                                {/* Model Selection */}
                                <div>
                                    <label className="text-sm font-medium text-slate-700 dark:text-slate-300 mb-2 block">比較モデル (複数選択可)</label>
                                    <div className="flex flex-wrap gap-2">
                                        {[
                                            { id: "v5_2025", label: "Model v5 (JRA Only)" },
                                            { id: "v4_2025", label: "Model v4 (JRA + 地方)" }
                                        ].map(m => (
                                            <button
                                                key={m.id}
                                                onClick={() => toggleModel(m.id)}
                                                className={`px-3 py-2 rounded-md border text-sm transition-all ${selectedModels.includes(m.id)
                                                    ? "bg-indigo-600 text-white border-indigo-600 shadow-md"
                                                    : "bg-white dark:bg-slate-800 border-slate-300 dark:border-slate-700 text-slate-600 dark:text-slate-300"
                                                    }`}
                                            >
                                                {m.label}
                                            </button>
                                        ))}
                                    </div>
                                </div>

                                {/* Year Selection */}
                                <div>
                                    <label className="text-sm font-medium text-slate-700 dark:text-slate-300 mb-2 block">対象期間 (Year)</label>
                                    <div className="flex gap-2">
                                        {[2024, 2025].map(y => (
                                            <button
                                                key={y}
                                                onClick={() => setSelectedYear(y)}
                                                className={`flex-1 py-1.5 rounded text-sm font-bold border transition-colors ${selectedYear === y
                                                    ? "bg-indigo-100 text-indigo-700 border-indigo-300 dark:bg-indigo-900/30 dark:text-indigo-300 dark:border-indigo-700"
                                                    : "bg-white dark:bg-slate-800 border-slate-200 dark:border-slate-700 text-slate-500"
                                                    }`}
                                            >
                                                {y}
                                            </button>
                                        ))}
                                    </div>
                                </div>

                                {/* Surface Filter */}
                                <div>
                                    <label className="text-sm font-medium text-slate-700 dark:text-slate-300 mb-2 block">馬場状態 (Surface)</label>
                                    <div className="flex rounded-md shadow-sm" role="group">
                                        {(["All", "芝", "ダート"] as const).map((s, idx) => (
                                            <button
                                                key={s}
                                                onClick={() => setSurfaceFilter(s)}
                                                className={`flex-1 py-1.5 text-sm border font-medium
                                                    ${idx === 0 ? "rounded-l-md" : ""} ${idx === 2 ? "rounded-r-md" : ""}
                                                    ${surfaceFilter === s
                                                        ? "bg-indigo-600 text-white border-indigo-600 z-10"
                                                        : "bg-white dark:bg-slate-800 border-slate-300 dark:border-slate-700 text-slate-600 -ml-px hover:bg-slate-50"
                                                    }`}
                                            >
                                                {s === "All" ? "すべて" : s}
                                            </button>
                                        ))}
                                    </div>
                                </div>
                            </CardContent>
                        </Card>

                        {/* Strategy Builder */}
                        <div>
                            <div className="flex items-center justify-between mb-2">
                                <h2 className="text-lg font-bold text-slate-800 dark:text-slate-200">戦略 (Strategies)</h2>
                                <Button size="sm" onClick={addStrategy} variant="outline" className="gap-1 border-dashed h-8">
                                    <Plus className="h-3 w-3" /> 追加
                                </Button>
                            </div>

                            <div className="space-y-4">
                                {strategies.map((strategy) => {
                                    const needsRow3 = ["SanRenTan", "SanRenPuku"].includes(strategy.type);
                                    const needsRow2 = needsRow3 || ["Umaren", "Wide", "Umatan"].includes(strategy.type);

                                    return (
                                        <Card key={strategy.id} className="relative overflow-hidden group border-slate-200 dark:border-slate-800 hover:border-indigo-300 transition-colors">
                                            <div className="absolute top-0 left-0 w-1 h-full bg-indigo-500"></div>
                                            <CardHeader className="py-2 px-3 bg-slate-50/50 dark:bg-slate-800/50 flex flex-row items-center justify-between space-y-0">
                                                <div className="flex items-center gap-2 flex-1">
                                                    <input
                                                        className="font-bold bg-transparent border-none focus:ring-0 p-0 text-slate-900 dark:text-white placeholder:text-slate-400 w-32"
                                                        value={strategy.name}
                                                        onChange={(e) => updateStrategy(strategy.id, { name: e.target.value })}
                                                        placeholder="名称未設定"
                                                    />
                                                    <select
                                                        className="bg-transparent text-xs text-slate-600 dark:text-slate-300 border-none focus:ring-0 cursor-pointer font-mono bg-slate-200/50 rounded px-1 py-0.5"
                                                        value={strategy.type}
                                                        onChange={(e) => updateStrategy(strategy.id, { type: e.target.value })}
                                                    >
                                                        {TICKET_TYPES.map(t => <option key={t.value} value={t.value}>{t.label}</option>)}
                                                    </select>
                                                </div>
                                                <button onClick={() => removeStrategy(strategy.id)} className="text-slate-400 hover:text-red-500">
                                                    <Trash2 className="h-4 w-4" />
                                                </button>
                                            </CardHeader>
                                            <CardContent className="p-3 space-y-2">
                                                {[0, 1, 2].map((rowIndex) => {
                                                    if (rowIndex === 1 && !needsRow2) return null;
                                                    if (rowIndex === 2 && !needsRow3) return null;
                                                    const label = rowIndex === 0 ? "1着/軸" : rowIndex === 1 ? "2着/相手" : "3着/紐";

                                                    return (
                                                        <div key={rowIndex} className="flex items-center gap-2">
                                                            <span className="w-14 text-[10px] font-bold text-slate-400 uppercase text-right">{label}</span>
                                                            <div className="flex flex-wrap gap-1">
                                                                {[1, 2, 3, 4, 5, 6].map((rank) => (
                                                                    <button
                                                                        key={rank}
                                                                        onClick={() => toggleRank(strategy.id, rowIndex, rank)}
                                                                        className={`w-6 h-6 rounded text-xs font-bold transition-all ${strategy.formation[rowIndex]?.includes(rank)
                                                                            ? "bg-indigo-600 text-white shadow-md"
                                                                            : "bg-slate-100 dark:bg-slate-800 text-slate-400 hover:bg-slate-200"
                                                                            }`}
                                                                    >
                                                                        {rank}
                                                                    </button>
                                                                ))}
                                                            </div>
                                                        </div>
                                                    );
                                                })}
                                            </CardContent>
                                        </Card>
                                    );
                                })}
                            </div>

                            {/* Action Button */}
                            <Button
                                onClick={runSimulation}
                                disabled={loading || strategies.length === 0 || selectedModels.length === 0}
                                className="w-full mt-6 bg-gradient-to-r from-indigo-600 to-violet-600 hover:from-indigo-700 hover:to-violet-700 text-white font-bold h-12 text-lg shadow-lg"
                            >
                                {loading ? (
                                    <div className="flex items-center gap-2">
                                        <div className="animate-spin rounded-full h-5 w-5 border-2 border-white/20 border-t-white"></div>
                                        <span>計算中...</span>
                                    </div>
                                ) : (
                                    <div className="flex items-center gap-2">
                                        <Play className="h-5 w-5 fill-current" />
                                        <span>シミュレーション実行</span>
                                    </div>
                                )}
                            </Button>

                            {error && (
                                <Alert variant="destructive" className="mt-4">
                                    <AlertTitle>エラー</AlertTitle>
                                    <AlertDescription>{error}</AlertDescription>
                                </Alert>
                            )}
                        </div>
                    </div>

                    {/* RIGHT COLUMN: Results */}
                    <div className="lg:col-span-7 space-y-8">
                        {result && Object.entries(result).map(([modelKey, strategiesResult]) => (
                            <div key={modelKey} className="space-y-4">
                                <div className="flex items-center gap-2 border-b border-slate-200 dark:border-slate-700 pb-2">
                                    <Badge variant="outline" className="text-lg px-3 py-1 bg-white dark:bg-slate-800">
                                        {modelKey === "v5_2025" ? "Model v5 (JRA Only)" : "Model v4 (JRA + 地方)"}
                                    </Badge>
                                </div>

                                {Object.entries(strategiesResult).map(([strategyName, stat]) => {
                                    // Handle API Error/Warning Responses
                                    if (strategyName === "error" || strategyName === "warning") {
                                        return (
                                            <Alert variant={strategyName === "error" ? "destructive" : "default"} key={strategyName} className="mb-4">
                                                <AlertTitle>{strategyName === "error" ? "Error" : "Warning"}</AlertTitle>
                                                <AlertDescription>{String(stat)}</AlertDescription>
                                            </Alert>
                                        );
                                    }

                                    // Type Guard
                                    if (typeof stat !== 'object' || !stat.summary) {
                                        return null;
                                    }

                                    return (
                                        <Card key={strategyName} className="border-l-4 border-l-indigo-500">
                                            <CardHeader className="py-3 bg-slate-50 dark:bg-slate-900/50">
                                                <CardTitle className="text-base flex justify-between items-center">
                                                    <span>{strategyName}</span>
                                                    <div className="flex gap-4 text-sm font-mono">
                                                        <div className="flex flex-col items-end">
                                                            <span className="text-xs text-slate-500">ROI</span>
                                                            <span className={`font-black text-lg ${stat.summary.roi >= 100 ? "text-green-600" : "text-red-500"}`}>
                                                                {stat.summary.roi.toFixed(1)}%
                                                            </span>
                                                        </div>
                                                        <div className="flex flex-col items-end">
                                                            <span className="text-xs text-slate-500">Profit</span>
                                                            <span className={`font-bold text-lg ${(stat.summary.total_profit ?? 0) >= 0 ? "text-green-600" : "text-red-500"}`}>
                                                                {(stat.summary.total_profit ?? 0) >= 0 ? "+" : ""}
                                                                {(stat.summary.total_profit ?? 0).toLocaleString()}
                                                            </span>
                                                        </div>
                                                    </div>
                                                </CardTitle>
                                            </CardHeader>
                                            <CardContent className="pt-4">
                                                {/* Summary Metrics */}
                                                <div className="grid grid-cols-3 gap-2 text-center text-sm mb-4">
                                                    <div className="bg-slate-100 dark:bg-slate-800 p-2 rounded">
                                                        <div className="text-xs text-slate-500">Total Bet</div>
                                                        <div className="font-mono">{stat.summary.total_bet.toLocaleString()}</div>
                                                    </div>
                                                    <div className="bg-slate-100 dark:bg-slate-800 p-2 rounded">
                                                        <div className="text-xs text-slate-500">Hit Rate</div>
                                                        <div className="font-mono">{stat.summary.hit_rate.toFixed(1)}%</div>
                                                    </div>
                                                    <div className="bg-slate-100 dark:bg-slate-800 p-2 rounded">
                                                        <div className="text-xs text-slate-500">Races</div>
                                                        <div className="font-mono">{stat.summary.race_count}</div>
                                                    </div>
                                                </div>

                                                {/* Drill Down Analysis */}
                                                <div className="mt-6">
                                                    <h4 className="text-sm font-bold text-slate-500 mb-2 flex items-center gap-1">
                                                        <TrendingUp className="h-4 w-4" /> 月別推移 (Monthly Analysis)
                                                    </h4>
                                                    <MonthlyBreakdown dailyStats={stat.daily} />
                                                </div>
                                            </CardContent>
                                        </Card>
                                    );
                                })}
                            </div>
                        ))}

                        {!result && !loading && (
                            <div className="h-full flex flex-col items-center justify-center text-slate-400 border-2 border-dashed border-slate-200 dark:border-slate-800 rounded-xl min-h-[400px]">
                                <Target className="h-16 w-16 mb-4 opacity-20" />
                                <p>左側のパネルで条件を設定し、<br />シミュレーションを実行してください。</p>
                            </div>
                        )}
                    </div>

                </div>
            </div>
        </div>
    );
}
