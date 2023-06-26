import LessonDetails from "./LessonDetails";
import { TabList, Tabs, Tab, TabPanel } from "react-tabs";
import "react-tabs/style/react-tabs.css";

function LessonsList() {
  return (
    <div className="h-full overflow-y-auto bg-slate-200">
      <LessonDetails />
      <LessonDetails />
      <LessonDetails />
      <LessonDetails />
      <LessonDetails />
      <LessonDetails />
      <LessonDetails />
      <LessonDetails />
      <LessonDetails />
    </div>
  );
}
export default LessonsList;
