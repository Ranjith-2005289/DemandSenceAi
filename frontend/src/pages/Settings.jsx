import { useCallback, useEffect, useState } from "react";
import { RefreshCw } from "lucide-react";
import PageHeader, { PageBody } from "@/components/shell/PageHeader";
import { Button, Chip, Facts, Panel, PanelBody, PanelHeader, Segmented, StatusDot } from "@/components/ui";
import { API_BASE_URL, errorMessage, getHealth } from "@/api";
import { useTheme } from "@/lib/theme";
import { MODEL_COUNT } from "@/lib/format";

export default function SettingsPage() {
  const { preference, setPreference } = useTheme();
  const [health, setHealth] = useState({ state: "checking", detail: null, at: null });

  const check = useCallback(async () => {
    setHealth((h) => ({ ...h, state: "checking" }));
    try {
      const res = await getHealth();
      setHealth({ state: res?.status === "ok" ? "ok" : "bad", detail: res?.service, at: new Date() });
    } catch (err) {
      setHealth({ state: "bad", detail: errorMessage(err), at: new Date() });
    }
  }, []);

  useEffect(() => {
    check();
  }, [check]);

  const status = {
    checking: <Chip>Checking…</Chip>,
    ok: (
      <Chip tone="ok">
        <StatusDot tone="ok" /> Connected
      </Chip>
    ),
    bad: (
      <Chip tone="bad">
        <StatusDot tone="bad" /> Unreachable
      </Chip>
    ),
  }[health.state];

  return (
    <>
      <PageHeader title="Settings" description="Preferences for this browser and the connection to the forecasting service." />
      <PageBody>
        <div className="max-w-3xl space-y-4">
        <Panel>
          <PanelHeader title="Appearance" subtitle="Saved in this browser." />
          <PanelBody className="flex items-center justify-between gap-4">
            <div>
              <p className="text-[13px] font-medium text-ink">Theme</p>
              <p className="text-xs text-ink-3">Light is the default. "Match system" follows your operating system.</p>
            </div>
            <Segmented
              value={preference}
              onChange={setPreference}
              options={[
                { value: "light", label: "Light" },
                { value: "dark", label: "Dark" },
                { value: "system", label: "Match system" },
              ]}
            />
          </PanelBody>
        </Panel>

        <Panel>
          <PanelHeader
            title="Forecasting service"
            subtitle="Set with VITE_API_BASE_URL when the app is deployed."
            actions={
              <Button size="sm" icon={RefreshCw} onClick={check} loading={health.state === "checking"}>
                Check again
              </Button>
            }
          />
          <PanelBody className="py-2">
            <Facts
              items={[
                ["Endpoint", API_BASE_URL],
                ["Status", status],
                health.state === "bad" && ["Detail", health.detail],
                health.at && ["Last checked", health.at.toLocaleTimeString()],
              ]}
            />
            <p className="mt-3 text-xs text-ink-3">
              Forecasting works without further setup. The Insights report and Assistant also need a Gemini API key in the server's
              environment (GEMINI_API_KEY).
            </p>
          </PanelBody>
        </Panel>

        <Panel>
          <PanelHeader title="About" />
          <PanelBody className="py-2">
            <Facts
              items={[
                ["Product", "DemandSense"],
                ["Version", "0.2.0"],
                ["Models", `${MODEL_COUNT} + top-3 ensemble`],
                ["Model selection", "MASE on a common holdout backtest"],
                ["Forecast range", "95%, calibrated on backtest errors"],
              ]}
            />
          </PanelBody>
        </Panel>
        </div>
      </PageBody>
    </>
  );
}
