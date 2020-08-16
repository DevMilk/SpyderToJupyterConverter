#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define MAX_LINE_LENGTH 500
#define MAX_FILE_NAME_LENGTH 200
#define IS_SPECIAL_CHARACTER(ch) !(((ch) >= 'a' && (ch) <= 'z') || ((ch) >= 'A' && (ch) <= 'Z') || (ch) >= '0' && (ch) <= '9')
  
  //Prefix for markdown type cells
  char markdownprefix[] = "{\n \
\"cell_type\": \"markdown\",\n \
\"metadata\": {},\n \
\"source\": [\n";
	
  //Prefix for code cells	
  char prefix[] = "{ \
\"cell_type\": \"code\",\n \
\"execution_count\": null,\n \
\"metadata\": {},\n \
\"outputs\": [],\n \
\"source\": [\n";
  
  //Prefix for cell start
  char start[] = "{\"cells\":[\n";
  
  //Cell end
  char end[] = "],\n \
\"metadata\": {\n \
\"kernelspec\": {\n \
\"display_name\": \"Python 3\",\n \
\"language\": \"python\",\n \
\"name\": \"python3\"\n \ \
},\n \
\"language_info\": {\n \
\"codemirror_mode\": {\n \
\"name\": \"ipython\",\n \
\"version\": 3\n \
},\n \
\"file_extension\": \".py\",\n \
\"mimetype\": \"text/x-python\",\n \
\"name\": \"python\",\n \
\"nbconvert_exporter\": \"python\",\n \
\"pygments_lexer\": \"ipython3\",\n \
\"version\": \"3.6.4\"\n \
}\n \
},\n \
\"nbformat\": 4,\n \
\"nbformat_minor\": 2}";

//Check if its a beginning of new cell 
int isNewCell(char * line) {
  return strstr(line, "#%%") != NULL;
}

//Get file name from given path
char *getFileName(char *path)
{
    char *aux = path;

    /* Go to end of string, so you don't need strlen */
    while (*path++) ;

    /* Find the last occurence of \ */
    while (*path-- != '\\' && path != aux) ;

    /* It must ignore the \ */
    return (aux == path) ? path : path + 2;
}
//Process lines for converting lines to valid form by handling quotes and special characters
char * processLine(char * line) {
  //Allocate new string	
  int processedLineLength = strlen(line) * 2;
  char * processed = (char * ) malloc(processedLineLength * sizeof(char));
  
  //Initialize counters
  int i = 0, j=0;
  for (i = 0; i < strlen(line); i++) {
    if (line[i] == '"' || line[i] == '\t'){
    	processed[j++] = "\\" [0];
    	if (line[i] == '\t')
    		processed[j++] = 't';
	}
    if (line[i] != '\n' && line[i] != '\t')
      processed[j++] = line[i];
  }
  processed[j++] = "\\" [0];
  processed[j++] = 'n';
  processed[j] = '\0';
  return processed;
}


int main(int argc, char* argv[]) {
	
  // If no argument is given, ask from user to enter a path 
  if(argc!=2){
  	argv[1] = (char*)malloc(sizeof(char)*MAX_FILE_NAME_LENGTH);
  	printf("Please enter Spyder Python Script Path: ");
  	scanf("%s",argv[1]);
  }
  //Spyder notebook file
  FILE * spyderNotebook = fopen(argv[1], "r");
  
  //New Jupyter notebook file
  FILE * jupyterNotebook;
  
  //counter, file content length 
  int i = 0, length = 0;
  
  //Flag for checking start of cell
  short isCellStart = 1;
  
  //Strings for keeping line and new Jupyter Notebook's name  
  char line[MAX_LINE_LENGTH],newFileName[MAX_FILE_NAME_LENGTH];
  
  //Content of Jupyter Notebook
  char * jupyter;
  
  //Get content length of Spyder Notebook
  while (fgetc(spyderNotebook)!=EOF) 
    i++;
  fclose(spyderNotebook);
  length = i;
	
  //Allocate jupyter content length using spyder notebook's length	
  jupyter = (char * ) malloc(sizeof(char) * (strlen(start) + strlen(end) + length * 2));
  
  //Reopen Spyder Notebook for parsing
  spyderNotebook = fopen(argv[1], "r");
  
  //Start of JSON
  strcat(jupyter, start);
  
  //Process Spyder Notebook to get content line by line and hande special characters 
  while (fgets(line, sizeof(line), spyderNotebook)) {
  	
  	//If new Code Cell, end previous cell, check for markdown
    if (isNewCell(line)) {
      //Code cell finish
      strcat(jupyter, "]\n},\n");
      
      //Check if a markdown available
      int k = 0;
      while (line[k++] != '%');
      int startMarkDownText = k+1;
      while (line[k++] != '\n');
      
	  //I
      if(k>4 ){
      	line[k-1]='\0';
        sprintf(jupyter, "%s%s\"%s\"]\n},\n", jupyter, markdownprefix, line+startMarkDownText);
      }
      // set isCellNew to 1 
      isCellStart = isCellStart | 1;
    } 
    //Add lines to source JSON array
	else {
	  //If it is beginning of a cell, concatenate prefix to content 	
      if (isCellStart) {
        //set zero by XOR (more optimize)
        isCellStart ^= isCellStart;
        strcat(jupyter, prefix);
      }
	  //If it is not beginning of a cell, add a coma for new line  
	  else
        strcat(jupyter, ",\n");
        
      //Add new line  
      sprintf(jupyter, "%s\"%s\"", jupyter, processLine(line));
    }

  }
  
  //Close spyderNotebook
  fclose(spyderNotebook);	
  
  //Close cells JSON array  
  strcat(jupyter, "]\n}\n");
  
  //Append ending prefix
  strcat(jupyter, end);
  
  //Get absolute file name and generate file name 
  i=0;
  argv[1] = getFileName(argv[1]);
  while(argv[1][i++]!='.');
  argv[1][i-1] = '\0';
  sprintf(newFileName,"JupyterOf-%s.ipynb",argv[1]);	
  printf("Created: %s\n",newFileName);	
  jupyterNotebook = fopen(newFileName, "w");
  
  //Bug Fix
  i=0;
  while(jupyter[i++]!='{');
  jupyter = (jupyter + i-1);
  
  //Check if new file created or not
  if (jupyterNotebook != NULL) {
  	printf("Done.");
    fputs(jupyter, jupyterNotebook);
    fclose(jupyterNotebook);
  }
  else{
  	printf("\nFile cannot be created!");
  	exit(0);
  }
  return 0;

}
