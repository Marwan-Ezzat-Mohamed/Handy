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

  if (format === "Sign") {
    return (
      <div className=" flex w-full flex-row px-2 py-5">
        <TextField
          id="input-with-icon-textfield"
          value=" Sign"
          InputProps={{
            readOnly: true,
            startAdornment: (
              <InputAdornment position="start">
                <SignLanguageIcon color="primary" />
              </InputAdornment>
            ),
          }}
          variant="outlined"
        />
        <IconButton color="primary" onClick={switchHandler}>
          <SwapHorizIcon />
        </IconButton>
        <TextField
          id="input-with-icon-textfield"
          value="Text"
          InputProps={{
            readOnly: true,
            startAdornment: (
              <InputAdornment position="start">
                <TextFormatIcon color="primary" fontSize="large" />
              </InputAdornment>
            ),
          }}
          variant="outlined"
        />
      </div>
    );
  } else {
    return (
      <div className=" flex w-full flex-row px-2 py-5">
        <TextField
          id="input-with-icon-textfield"
          value="Text"
          InputProps={{
            readOnly: true,
            startAdornment: (
              <InputAdornment position="start">
                <TextFormatIcon color="primary" fontSize="large" />
              </InputAdornment>
            ),
          }}
          variant="outlined"
        />
        <IconButton color="primary" onClick={switchHandler}>
          <SwapHorizIcon />
        </IconButton>
        <TextField
          id="input-with-icon-textfield"
          value=" Sign"
          InputProps={{
            readOnly: true,
            startAdornment: (
              <InputAdornment position="start">
                <SignLanguageIcon color="primary" />
              </InputAdornment>
            ),
          }}
          variant="outlined"
        />
      </div>
    );
  }
}
export default FormatBar;
