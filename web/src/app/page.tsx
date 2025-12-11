"use client";

import { useState, useEffect } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { ModeToggle } from "@/components/mode-toggle";

interface Race {
  race_id: string;
  venue: string;
  race_number: number;
  title: string;
  start_time: string;
  distance?: number;
  surface?: string;
  weather?: string;
  state?: string;
}

// JRA Venue Code Mapping
const VENUE_MAP: Record<string, string> = {
  "01": "札幌 (Sapporo)",
  "02": "函館 (Hakodate)",
  "03": "福島 (Fukushima)",
  "04": "新潟 (Niigata)",
  "05": "東京 (Tokyo)",
  "06": "中山 (Nakayama)",
  "07": "中京 (Chukyo)",
  "08": "京都 (Kyoto)",
  "09": "阪神 (Hanshin)",
  "10": "小倉 (Kokura)",
  // Add common NAR codes if needed in future
};

export default function Home() {
  const [races, setRaces] = useState<Race[]>([]);
  const [loading, setLoading] = useState(false);
  const [selectedDate, setSelectedDate] = useState<string>(""); // Will be set from API
  const [prevDate, setPrevDate] = useState<string | null>(null);
  const [nextDate, setNextDate] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [dailyRoi, setDailyRoi] = useState<any>(null);

  const fetchRaces = async (date: string) => {
    if (!date) return;
    setLoading(true);
    setError(null);
    try {
      const response = await fetch(`http://localhost:8000/api/races?date=${date}`);
      if (!response.ok) {
        throw new Error("Failed to fetch races");
      }
      const data = await response.json();
      setRaces(data.races || []);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    // Fetch the latest race date from API
    const fetchLatestDate = async () => {
      try {
        const res = await fetch("http://localhost:8000/api/races/latest-date");
        if (res.ok) {
          const data = await res.json();
          if (data.latest_date) {
            setSelectedDate(data.latest_date);
            return;
          }
        }
      } catch (e) {
        console.warn("Failed to fetch latest date, using fallback");
      }
      // Fallback to a known date with data
      setSelectedDate("2024-12-07");
    };

    fetchLatestDate();
  }, []);

  useEffect(() => {
    if (selectedDate) {
      fetchRaces(selectedDate);

      // Fetch adjacent dates
      fetch(`http://localhost:8000/api/races/adjacent-dates?date=${selectedDate}`)
        .then(res => res.json())
        .then(data => {
          setPrevDate(data.prev_date || null);
          setNextDate(data.next_date || null);
        })
        .catch(() => {
          setPrevDate(null);
          setNextDate(null);
        });

      // Fetch daily ROI
      fetch(`http://localhost:8000/api/daily-roi?date=${selectedDate}`)
        .then(res => res.json())
        .then(data => {
          if (!data.error) {
            setDailyRoi(data);
          } else {
            setDailyRoi(null);
          }
        })
        .catch(() => setDailyRoi(null));
    }
  }, [selectedDate]);

  // Group races by venue (Filtering out non-JRA/unmapped venues)
  const groupedRaces = races.reduce((acc, race) => {
    // Check if it's a known JRA venue
    if (!VENUE_MAP[race.venue]) {
      // Skip NRA/NAR or unknown venues as requested
      return acc;
    }

    const venueName = VENUE_MAP[race.venue];
    if (!acc[venueName]) {
      acc[venueName] = [];
    }
    acc[venueName].push(race);
    return acc;
  }, {} as Record<string, Race[]>);

  const sortedVenueNames = Object.keys(groupedRaces).sort((a, b) => {
    // Sort mapping to keep standard JRA order (North to South/East to West approx)
    // This is a rough heuristic since we only have names now. 
    // Ideally we sort by the original code.
    const codeA = Object.keys(VENUE_MAP).find(key => VENUE_MAP[key] === a) || "99";
    const codeB = Object.keys(VENUE_MAP).find(key => VENUE_MAP[key] === b) || "99";
    return codeA.localeCompare(codeB);
  });

  return (
    <div className="min-h-screen bg-slate-50 dark:bg-slate-900 text-slate-900 dark:text-slate-100 font-sans selection:bg-indigo-500 selection:text-white">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="mb-10 flex flex-col md:flex-row md:items-end justify-between gap-4 border-b border-slate-200 dark:border-slate-800 pb-6">
          <div>
            <h1 className="text-3xl md:text-4xl font-extrabold tracking-tight mb-2 bg-clip-text text-transparent bg-gradient-to-r from-indigo-600 to-violet-600 dark:from-indigo-400 dark:to-violet-400">
              Keiiba-AI Dashboard
            </h1>
            <p className="text-slate-500 dark:text-slate-400 font-medium">
              Next-Gen Horse Racing Prediction System
            </p>
          </div>

          {/* Actions & Date Selector */}
          <div className="flex flex-col sm:flex-row items-stretch sm:items-center gap-3">
            <ModeToggle />
            <Button
              onClick={() => window.location.href = '/simulation'}
              variant="outline"
              className="bg-white dark:bg-slate-800 border-slate-200 dark:border-slate-700 hover:bg-slate-50 hover:text-indigo-600 font-bold"
            >
              <span className="mr-2">🚀</span> Betting Simulator
            </Button>

            <div className="flex items-center gap-2 bg-white dark:bg-slate-800 p-2 rounded-lg shadow-sm border border-slate-200 dark:border-slate-700">
              {/* Previous Date Button */}
              <Button
                onClick={() => prevDate && setSelectedDate(prevDate)}
                size="sm"
                variant="ghost"
                disabled={!prevDate}
                title={prevDate ? `前の開催日: ${prevDate}` : "前の開催日なし"}
              >
                ←
              </Button>

              <span className="text-sm font-bold text-slate-500">DATE</span>
              <input
                type="date"
                value={selectedDate}
                onChange={(e) => setSelectedDate(e.target.value)}
                className="px-3 py-1 bg-transparent border-none focus:ring-0 font-mono text-sm"
              />

              {/* Next Date Button */}
              <Button
                onClick={() => nextDate && setSelectedDate(nextDate)}
                size="sm"
                variant="ghost"
                disabled={!nextDate}
                title={nextDate ? `次の開催日: ${nextDate}` : "次の開催日なし"}
              >
                →
              </Button>
            </div>
          </div>
        </div>

        {/* Daily ROI Summary */}
        {dailyRoi && dailyRoi.total && (
          <div className="mb-6 p-4 bg-white dark:bg-slate-800 rounded-lg border shadow-sm">
            <div className="flex flex-wrap items-center gap-6">
              {/* Total ROI */}
              <div className="flex items-center gap-3">
                <span className="text-sm font-bold text-slate-500">📊 当日推奨買い目</span>
                <div className={`px-3 py-1 rounded-full font-bold ${dailyRoi.total.roi >= 100
                  ? 'bg-green-100 text-green-700 dark:bg-green-900 dark:text-green-300'
                  : 'bg-red-100 text-red-700 dark:bg-red-900 dark:text-red-300'
                  }`}>
                  ROI {dailyRoi.total.roi}%
                </div>
                <span className="text-sm">
                  投資 ¥{dailyRoi.total.cost.toLocaleString()} →
                  払戻 ¥{dailyRoi.total.return.toLocaleString()}
                  <span className={dailyRoi.total.profit >= 0 ? 'text-green-600' : 'text-red-600'}>
                    {' '}({dailyRoi.total.profit >= 0 ? '+' : ''}¥{dailyRoi.total.profit.toLocaleString()})
                  </span>
                </span>
                <span className="text-xs text-slate-400">
                  {dailyRoi.total.races}R / 的中率 {dailyRoi.total.hit_rate}%
                </span>
              </div>

              {/* Per Venue */}
              {dailyRoi.by_venue && dailyRoi.by_venue.length > 0 && (
                <div className="flex items-center gap-2 border-l pl-4 border-slate-200 dark:border-slate-700">
                  <span className="text-xs text-slate-400">競馬場別:</span>
                  {dailyRoi.by_venue.map((v: any) => (
                    <span
                      key={v.venue}
                      className={`text-xs px-2 py-0.5 rounded ${v.roi >= 100
                        ? 'bg-green-50 text-green-600 dark:bg-green-900/30 dark:text-green-400'
                        : 'bg-slate-100 text-slate-500 dark:bg-slate-700 dark:text-slate-400'
                        }`}
                      title={`¥${v.cost.toLocaleString()} → ¥${v.return.toLocaleString()} (${v.hit_rate}%的中)`}
                    >
                      {VENUE_MAP[v.venue]?.split(' ')[0] || v.venue} {v.roi}%
                    </span>
                  ))}
                </div>
              )}
            </div>
          </div>
        )}

        {/* Race List */}
        {loading && (
          <div className="flex flex-col items-center justify-center py-20">
            <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-indigo-600 mb-4"></div>
            <p className="text-slate-500 animate-pulse">Loading race schedule...</p>
          </div>
        )}

        {error && (
          <div className="p-4 rounded-lg bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 text-red-600 dark:text-red-400">
            Error: {error}
          </div>
        )}

        {!loading && !error && races.length === 0 && (
          <div className="text-center py-20 text-slate-400">
            <p>No races found for this date.</p>
          </div>
        )}

        {!loading && !error && Object.keys(groupedRaces).length > 0 && (
          <div className="flex flex-col md:flex-row gap-6 overflow-x-auto pb-8 snap-x">
            {sortedVenueNames.map((venueName) => (
              <div key={venueName} className="flex-1 min-w-[280px] max-w-sm flex flex-col gap-4 snap-start">
                {/* Venue Header with ROI */}
                {(() => {
                  // Find venue code from name
                  const venueCode = Object.keys(VENUE_MAP).find(k => VENUE_MAP[k] === venueName) || '';
                  const venueRoi = dailyRoi?.by_venue?.find((v: any) => v.venue === venueCode);
                  return (
                    <div className="sticky top-0 z-10 bg-slate-50/95 dark:bg-slate-900/95 backdrop-blur-sm py-2 border-b-2 border-indigo-500">
                      <div className="flex justify-between items-center">
                        <h2 className="text-lg font-bold text-slate-800 dark:text-slate-100 flex items-center gap-2">
                          <span className="inline-block w-2 h-2 rounded-full bg-indigo-500"></span>
                          {venueName}
                        </h2>
                        {venueRoi && (
                          <div className="flex items-center gap-2 text-xs">
                            <span className={`px-2 py-0.5 rounded font-bold ${venueRoi.roi >= 100
                              ? 'bg-green-100 text-green-700 dark:bg-green-900 dark:text-green-300'
                              : 'bg-red-100 text-red-700 dark:bg-red-900 dark:text-red-300'
                              }`}>
                              {venueRoi.roi}%
                            </span>
                            <span className="text-slate-400">
                              ¥{venueRoi.cost.toLocaleString()}→¥{venueRoi.return.toLocaleString()}
                            </span>
                          </div>
                        )}
                      </div>
                    </div>
                  );
                })()}

                {/* Race Cards for this Venue */}
                <div className="flex flex-col gap-3">
                  {groupedRaces[venueName]
                    .sort((a, b) => a.race_number - b.race_number)
                    .map((race) => (
                      <div
                        key={race.race_id}
                        className="group relative bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 p-4 hover:shadow-lg hover:border-indigo-400 dark:hover:border-indigo-500 transition-all duration-200 cursor-pointer"
                        onClick={() => window.location.href = `/races/${race.race_id}`}
                      >
                        <div className="flex justify-between items-start mb-2">
                          <div className="flex items-center gap-2">
                            <span className="flex items-center justify-center w-8 h-8 rounded-lg bg-slate-100 dark:bg-slate-700 text-slate-900 dark:text-white font-bold text-sm group-hover:bg-indigo-100 dark:group-hover:bg-indigo-900 group-hover:text-indigo-700 dark:group-hover:text-indigo-300 transition-colors">
                              {race.race_number}R
                            </span>
                            <span className="text-xs font-mono font-medium text-slate-500 border border-slate-200 dark:border-slate-600 rounded px-1.5 py-0.5 bg-slate-50 dark:bg-slate-800">
                              {race.start_time}
                            </span>
                          </div>
                          <div className="flex gap-1">
                            {race.surface && (
                              <span className={`text-[10px] px-1.5 py-0.5 rounded font-bold ${race.surface === '芝'
                                ? 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400'
                                : race.surface === 'ダート'
                                  ? 'bg-amber-100 text-amber-700 dark:bg-amber-900/30 dark:text-amber-400'
                                  : 'bg-slate-100 text-slate-600'
                                }`}>
                                {race.surface}
                              </span>
                            )}
                            {race.distance && (
                              <span className="text-[10px] px-1.5 py-0.5 rounded bg-slate-100 dark:bg-slate-700 text-slate-600 dark:text-slate-300 font-mono">
                                {race.distance}m
                              </span>
                            )}
                          </div>
                        </div>
                        <h3 className="font-bold text-slate-800 dark:text-slate-200 group-hover:text-indigo-600 dark:group-hover:text-indigo-400 transition-colors line-clamp-1 mb-1">
                          {race.title || "Race Name N/A"}
                        </h3>
                        <div className="flex items-center gap-2 text-xs text-slate-500 dark:text-slate-400">
                          {race.weather && <span>{race.weather}</span>}
                          {race.state && (
                            <span className={`px-1 rounded ${['良'].includes(race.state) ? 'bg-blue-50 text-blue-600 dark:bg-blue-900/20 dark:text-blue-400' :
                              ['稍重'].includes(race.state) ? 'bg-yellow-50 text-yellow-600 dark:bg-yellow-900/20 dark:text-yellow-400' :
                                ['重', '不良'].includes(race.state) ? 'bg-red-50 text-red-600 dark:bg-red-900/20 dark:text-red-400' : ''
                              }`}>
                              {race.state}
                            </span>
                          )}
                        </div>

                        {/* ROI per race */}
                        {dailyRoi?.by_race?.[race.race_id] && (
                          <div className="mt-2 pt-2 border-t border-slate-100 dark:border-slate-700 flex justify-between items-center text-xs">
                            <span className="text-slate-400">
                              {dailyRoi.by_race[race.race_id].bet_type}
                            </span>
                            <span className={dailyRoi.by_race[race.race_id].hit
                              ? 'text-green-600 font-bold'
                              : 'text-slate-400'
                            }>
                              ¥{dailyRoi.by_race[race.race_id].cost.toLocaleString()} →
                              ¥{dailyRoi.by_race[race.race_id].return.toLocaleString()}
                              {dailyRoi.by_race[race.race_id].hit && ' ◎'}
                            </span>
                          </div>
                        )}

                        <div className="mt-3 flex justify-end">
                          <span className="text-xs font-semibold text-indigo-500 opacity-0 group-hover:opacity-100 transition-opacity flex items-center gap-1">
                            View Analysis <span className="text-lg leading-none">›</span>
                          </span>
                        </div>
                      </div>
                    ))}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
