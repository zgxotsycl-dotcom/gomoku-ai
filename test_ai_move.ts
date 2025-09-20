async function testGetAiMove() {
  const supabaseUrl = "https://xkwgfidiposftwwasdqs.supabase.co"; // The project URL from the deploy output
  const functionName = "get-ai-move";
  // This is the only line you should have edited. It is now correct.
  const anonKey = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Inhrd2dmaWRpcG9zZnR3d2FzZHFzIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTUwODM3NzMsImV4cCI6MjA3MDY1OTc3M30.-9n_26ga07dXFiFOShP78_p9cEcIKBxHBEYJ1A1gaiE"; 

  

  const url = `${supabaseUrl}/functions/v1/${functionName}`;

  const headers = {
    "Authorization": `Bearer ${anonKey}`,
    "Content-Type": "application/json",
  };

  const size = 15;
  const emptyRow = Array(size).fill(null);
  const emptyBoard = Array(size).fill(null).map(() => emptyRow.slice());
  // place two stones near center
  const mid = Math.floor(size / 2);
  emptyBoard[mid - 1][mid - 1] = "black" as const;
  emptyBoard[mid][mid] = "white" as const;

  const body = {
    board: emptyBoard,
    player: "black",
    moves: [
      [mid - 1, mid - 1],
      [mid, mid],
    ],
  };

  try {
    console.log(`Sending request to ${url}...`);
    const response = await fetch(url, {
      method: "POST",
      headers: headers,
      body: JSON.stringify(body),
    });

    console.log(`Response Status: ${response.status}`);
    
    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Request failed: ${response.statusText} - ${errorText}`);
    }

    const data = await response.json();
    console.log("Received data:");
    console.log(JSON.stringify(data, null, 2));

  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : String(error);
    console.error("An error occurred:", errorMessage);
  }
}

// Run the test
testGetAiMove();
