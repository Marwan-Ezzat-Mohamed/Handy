import React, { useState } from "react";
import TextField from "@mui/material/TextField";
import SwapHorizIcon from "@mui/icons-material/SwapHoriz";
import SignLanguageIcon from "@mui/icons-material/SignLanguage";
import TextFormatIcon from "@mui/icons-material/TextFormat";
import { IconButton, InputAdornment } from "@mui/material";
function FormatBar() {
  const [label, setLabel] = useState<string>("Sign");
  return (
    <div className=" flex w-full flex-row">
      <TextField
        id="input-with-icon-textfield"
        defaultValue={label === "Sign" ? "Sign" : "Text"}
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
      <IconButton color="primary" aria-label="add to shopping cart">
        <SwapHorizIcon />
      </IconButton>
      <TextField
        id="input-with-icon-textfield"
        defaultValue={label === "Sign" ? "Text" : "Sign"}
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
}
export default FormatBar;
