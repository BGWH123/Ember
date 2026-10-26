import { create } from 'zustand';
import type { SubmissionResult, CustomTest, AiHelpConfig, SubmissionHistory } from '@/lib/types';

type BottomTab = 'testcases' | 'testresults' | 'submissions';
type LeftTab = 'description' | 'solution' | 'aiHelp';

interface WorkspaceSession {
  currentCode: string;
  submissionResult: SubmissionResult | null;
  bottomTab: BottomTab;
  leftTab: LeftTab;
  selectedCaseIndex: number;
  customTests: CustomTest[];
  runResult: SubmissionResult | null;
  submissionHistory: SubmissionHistory[];
  aiHelpResponse: string | null;
  aiHelpError: string | null;
  horizontalRatio: number;
  verticalRatio: number;
}

function freshSession(): WorkspaceSession {
  return {
    currentCode: '',
    submissionResult: null,
    bottomTab: 'testcases',
    leftTab: 'description',
    selectedCaseIndex: 0,
    customTests: [],
    runResult: null,
    submissionHistory: [],
    aiHelpResponse: null,
    aiHelpError: null,
    horizontalRatio: 0.5,
    verticalRatio: 0.65,
  };
}

function captureSession(state: ProblemStore): WorkspaceSession {
  return {
    currentCode: state.currentCode,
    submissionResult: state.submissionResult,
    bottomTab: state.bottomTab,
    leftTab: state.leftTab,
    selectedCaseIndex: state.selectedCaseIndex,
    customTests: state.customTests,
    runResult: state.runResult,
    submissionHistory: state.submissionHistory,
    aiHelpResponse: state.aiHelpResponse,
    aiHelpError: state.aiHelpError,
    horizontalRatio: state.horizontalRatio,
    verticalRatio: state.verticalRatio,
  };
}

interface ProblemStore {
  activeProblemId: string | null;
  sessions: Record<string, WorkspaceSession>;
  activateProblem: (problemId: string) => boolean;
  currentCode: string;
  setCurrentCode: (code: string) => void;
  submissionResult: SubmissionResult | null;
  setSubmissionResult: (result: SubmissionResult | null) => void;
  isSubmitting: boolean;
  setIsSubmitting: (v: boolean) => void;
  drawerOpen: boolean;
  setDrawerOpen: (v: boolean) => void;
  // test panel
  bottomTab: BottomTab;
  setBottomTab: (tab: BottomTab) => void;
  leftTab: LeftTab;
  setLeftTab: (tab: LeftTab) => void;
  selectedCaseIndex: number;
  setSelectedCaseIndex: (i: number) => void;
  customTests: CustomTest[];
  addCustomTest: (test: CustomTest) => void;
  removeCustomTest: (index: number) => void;
  updateCustomTest: (index: number, test: CustomTest) => void;
  isRunning: boolean;
  setIsRunning: (v: boolean) => void;
  runResult: SubmissionResult | null;
  setRunResult: (r: SubmissionResult | null) => void;
  submissionHistory: SubmissionHistory[];
  setSubmissionHistory: (h: SubmissionHistory[]) => void;
  aiHelpConfig: AiHelpConfig;
  setAiHelpConfig: (patch: Partial<AiHelpConfig>) => void;
  aiHelpConfigOpen: boolean;
  setAiHelpConfigOpen: (value: boolean) => void;
  aiHelpCustomPrompt: string;
  setAiHelpCustomPrompt: (value: string) => void;
  aiHelpResponse: string | null;
  setAiHelpResponse: (response: string | null) => void;
  aiHelpError: string | null;
  setAiHelpError: (error: string | null) => void;
  aiHelpLoading: boolean;
  setAiHelpLoading: (value: boolean) => void;
  horizontalRatio: number;
  setHorizontalRatio: (value: number) => void;
  verticalRatio: number;
  setVerticalRatio: (value: number) => void;
  resetAiHelp: () => void;
  resetTestPanel: () => void;
}

export const useProblemStore = create<ProblemStore>((set, get) => ({
  activeProblemId: null,
  sessions: {},
  activateProblem: (problemId) => {
    const state = get();
    if (state.activeProblemId === problemId) return true;
    let sessions = state.sessions;
    if (state.activeProblemId) {
      sessions = {
        ...sessions,
        [state.activeProblemId]: captureSession(state),
      };
    }
    const cached = sessions[problemId];
    set({
      ...(cached ?? freshSession()),
      activeProblemId: problemId,
      sessions,
      isRunning: false,
      isSubmitting: false,
      aiHelpLoading: false,
    });
    return Boolean(cached);
  },
  ...freshSession(),
  setCurrentCode: (code) => set({ currentCode: code }),
  submissionResult: null,
  setSubmissionResult: (result) => set({ submissionResult: result }),
  isSubmitting: false,
  setIsSubmitting: (v) => set({ isSubmitting: v }),
  drawerOpen: false,
  setDrawerOpen: (v) => set({ drawerOpen: v }),
  // test panel
  bottomTab: 'testcases',
  setBottomTab: (tab) => set({ bottomTab: tab }),
  leftTab: 'description',
  setLeftTab: (tab) => set({ leftTab: tab }),
  selectedCaseIndex: 0,
  setSelectedCaseIndex: (i) => set({ selectedCaseIndex: i }),
  customTests: [],
  addCustomTest: (test) => set((s) => ({ customTests: [...s.customTests, test] })),
  removeCustomTest: (index) => set((s) => ({ customTests: s.customTests.filter((_, i) => i !== index) })),
  updateCustomTest: (index, test) => set((s) => {
    const next = [...s.customTests];
    next[index] = test;
    return { customTests: next };
  }),
  isRunning: false,
  setIsRunning: (v) => set({ isRunning: v }),
  runResult: null,
  setRunResult: (r) => set({ runResult: r }),
  submissionHistory: [],
  setSubmissionHistory: (h) => set({ submissionHistory: h }),
  aiHelpConfig: {
    baseUrl: '',
    apiKey: '',
    model: '',
    includeUserCode: false,
  },
  setAiHelpConfig: (patch) => set((s) => ({ aiHelpConfig: { ...s.aiHelpConfig, ...patch } })),
  aiHelpConfigOpen: false,
  setAiHelpConfigOpen: (value) => set({ aiHelpConfigOpen: value }),
  aiHelpCustomPrompt: '',
  setAiHelpCustomPrompt: (value) => set({ aiHelpCustomPrompt: value }),
  aiHelpResponse: null,
  setAiHelpResponse: (response) => set({ aiHelpResponse: response }),
  aiHelpError: null,
  setAiHelpError: (error) => set({ aiHelpError: error }),
  aiHelpLoading: false,
  setAiHelpLoading: (value) => set({ aiHelpLoading: value }),
  horizontalRatio: 0.5,
  setHorizontalRatio: (value) => set({ horizontalRatio: value }),
  verticalRatio: 0.65,
  setVerticalRatio: (value) => set({ verticalRatio: value }),
  resetAiHelp: () => set({ aiHelpResponse: null, aiHelpError: null, aiHelpLoading: false }),
  resetTestPanel: () => set({ bottomTab: 'testcases', selectedCaseIndex: 0, customTests: [], runResult: null, submissionHistory: [] }),
}));
