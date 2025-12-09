"use client";

import { useState, useEffect } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";

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
  const [selectedDate, setSelectedDate] = useState("2024-12-07"); // Updated default to generic date or today
  const [error, setError] = useState<string | null>(null);

  const fetchRaces = async (date: string) => {
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
    const today = new Date().toISOString().split('T')[0];
    // If user wants to default to today, uncomment next line. 
    // Keeping hardcoded for demo consistency unless requested.
    // fetchRaces(today);
    fetchRaces(selectedDate);
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
            <Button
              onClick={() => window.location.href = '/simulation'}
              variant="outline"
              className="bg-white dark:bg-slate-800 border-slate-200 dark:border-slate-700 hover:bg-slate-50 hover:text-indigo-600 font-bold"
            >
              <span className="mr-2">🚀</span> Betting Simulator
            </Button>

            <div className="flex items-center gap-2 bg-white dark:bg-slate-800 p-2 rounded-lg shadow-sm border border-slate-200 dark:border-slate-700">
              <span className="text-sm font-bold text-slate-500 px-2">DATE</span>
              <input
                type="date"
                value={selectedDate}
                onChange={(e) => setSelectedDate(e.target.value)}
                className="px-3 py-1 bg-transparent border-none focus:ring-0 font-mono text-sm"
              />
              <Button onClick={() => fetchRaces(selectedDate)} size="sm" variant="default">
                Refresh
              </Button>
            </div>
          </div>
        </div>

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
                {/* Venue Header */}
                <div className="sticky top-0 z-10 bg-slate-50/95 dark:bg-slate-900/95 backdrop-blur-sm py-2 border-b-2 border-indigo-500">
                  <h2 className="text-lg font-bold text-slate-800 dark:text-slate-100 flex items-center gap-2">
                    <span className="inline-block w-2 h-2 rounded-full bg-indigo-500"></span>
                    {venueName}
                  </h2>
                </div>

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
