import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import TextField from "@mui/material/TextField";
import SwapHorizIcon from "@mui/icons-material/SwapHoriz";
import SignLanguageIcon from "@mui/icons-material/SignLanguage";
import TextFormatIcon from "@mui/icons-material/TextFormat";
import { IconButton, InputAdornment } from "@mui/material";
function FormatBar() {
  const [format, setFormat] = useState<string>("Sign");
  const navigate = useNavigate();
  const switchHandler = () => {
    if (format === "Sign") {
      setFormat("Text");
      navigate("/texttosign");
    } else {
      setFormat("Sign");
      navigate("/");
    }
  };

  const arr = [
    <TextField
      style={{
        border: "1px solid #683aff",
        borderRadius: "6px",
      }}
      id="input-with-icon-textfield"
      value=""
      key="Sign"
      InputProps={{
        readOnly: true,
        startAdornment: (
          <InputAdornment
            style={{ fontWeight: "bold", color: "#683aff" }}
            position="start"
          >
            <SignLanguageIcon />
            Sign
          </InputAdornment>
        ),
      }}
      variant="outlined"
    />,
    <TextField
      style={{
        border: "1px solid #683aff",
        borderRadius: "6px",
      }}
      id="input-with-icon-textfield"
      value=""
      key="Text"
      InputProps={{
        readOnly: true,
        startAdornment: (
          <InputAdornment
            style={{ fontWeight: "bold", color: "#683aff" }}
            position="start"
          >
            <TextFormatIcon fontSize="large" />
            Text
          </InputAdornment>
        ),
      }}
      variant="outlined"
    />,
  ];

  const first = format === "Sign" ? 0 : 1;
  const second = format === "Sign" ? 1 : 0;

  return (
    <div className="flex w-full flex-row justify-center px-2 py-5">
      <div className="flex">
        {arr[first]}
        <IconButton style={{ color: "#683aff" }} onClick={switchHandler}>
          <SwapHorizIcon />
        </IconButton>
        {arr[second]}
      </div>
    </div>
  );
}
export default FormatBar;
