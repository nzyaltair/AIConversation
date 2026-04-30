import {
  createBrowserRouter,
  Navigate,
  Outlet,
} from 'react-router-dom';
import { RootLayout } from '@/layouts/root-layout';
import { ConversationPage } from '@/pages/conversation-page';
import { ConversationApiPage } from '@/pages/conversation-api-page';
import { ChatPage } from '@/pages/chat-page';
import { SpeechToTextPage } from '@/pages/speech-to-text-page';
import { TextToSpeechPage } from '@/pages/text-to-speech-page';
import { ModelsPage } from '@/pages/models-page';
import { VadPage } from '@/pages/vad-page';
import { NotFoundPage } from '@/pages/not-found-page';

export const router = createBrowserRouter([
  {
    element: (
      <RootLayout>
        <Outlet />
      </RootLayout>
    ),
    children: [
      { path: '/', element: <Navigate to="/models" replace /> },
      { path: '/conversation', element: <ConversationPage /> },
      { path: '/conversation-API', element: <ConversationApiPage /> },
      { path: '/chat', element: <ChatPage /> },
      { path: '/speech-to-text', element: <SpeechToTextPage /> },
      { path: '/text-to-speech', element: <TextToSpeechPage /> },
      { path: '/vad', element: <VadPage /> },
      { path: '/models', element: <ModelsPage /> },
      { path: '/voice', element: <Navigate to="/conversation" replace /> },
      { path: '/my-models', element: <Navigate to="/models" replace /> },
      { path: '*', element: <NotFoundPage /> },
    ],
  },
]);
