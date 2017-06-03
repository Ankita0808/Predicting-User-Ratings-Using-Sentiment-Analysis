/*
 * Copyright 2013 Alex Holmes
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */


import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IOUtils;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;

import edu.stanford.nlp.io.*;
import edu.stanford.nlp.ling.*;
import edu.stanford.nlp.pipeline.*;
import edu.stanford.nlp.sentiment.SentimentCoreAnnotations;
import edu.stanford.nlp.trees.*;
import edu.stanford.nlp.util.*;

import java.util.*;
import java.io.IOException;
import java.io.OutputStreamWriter;

/*
import com.fasterxml.jackson.core.JsonFactory;
import com.fasterxml.jackson.core.JsonParseException;
import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.core.JsonToken;
*/

//import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
//import org.json.simple.JSONValue;
import org.json.simple.parser.JSONParser;
import org.json.simple.parser.ParseException;

/**
 * An example MapReduce job showing how to use the {@link com.alexholmes.json.mapreduce.MultiLineJsonInputFormat}.
 */
public final class ExampleJob extends Configured implements Tool {

    /**
     * Main entry point for the example.
     *
     * @param args arguments
     * @throws Exception when something goes wrong
     */
     
    public static void main(final String[] args) throws Exception {
        int res = ToolRunner.run(new Configuration(), new ExampleJob(), args);
        System.exit(res);
    }

    public int run(final String[] args) throws Exception {

        String input = args[0];
        String output = args[1];
		
        Configuration conf = super.getConf();

        Job job = new Job(conf);
        job.setJarByClass(ExampleJob.class);
        job.setMapperClass(Map.class);
        //job.setNumReduceTasks(0);
		job.setCombinerClass(ExampleJob.RatingReducer.class);
    	job.setReducerClass(ExampleJob.RatingReducer.class);
        Path outputPath = new Path(output);
		
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(DoubleWritable.class);
        
        FileInputFormat.setInputPaths(job, input);
        FileOutputFormat.setOutputPath(job, outputPath);

        // use the JSON input format
        job.setInputFormatClass(MultiLineJsonInputFormat.class);
        //job.setOutputFormatClass(TextOutputFormat.class);

        // specify the JSON attribute name which is used to determine which
        // JSON elements are supplied to the mapper
        MultiLineJsonInputFormat.setInputJsonMember(job, "business_id");
		
        if (job.waitForCompletion(true)) {
            return 0;
        }
        return 1;
    }

    /**
     * JSON objects are supplied in string form to the mapper.
     * Here we are simply emitting them for viewing on HDFS.
     */
    private static class Map extends Mapper<LongWritable, Text, Text, DoubleWritable> {
    
		public Properties instantiatePipeline() {
			Properties props = new Properties();
			props.setProperty("annotators", "tokenize, ssplit, pos, lemma, parse, sentiment");
			props.setProperty("parser.maxlen", "50");
			return props;
		}	
			final StanfordCoreNLP pipeline = new StanfordCoreNLP(instantiatePipeline());
			static DoubleWritable rate;

		    @Override
	    	protected void map(LongWritable key, Text value, Context context)
	            throws IOException, InterruptedException {

		 try {
			JSONParser jsonParser = new JSONParser();
			JSONObject jsonObject = (JSONObject) jsonParser.parse(value.toString());
			String bus_id = (String) jsonObject.get("business_id"); 
		 	String review = (String) jsonObject.get("text");
			//context.write(new Text(review), null);
			review = review.replace('\n', ' ');
			
			Annotation annotation = pipeline.process(review);
		

			List<CoreMap> sentences = annotation.get(CoreAnnotations.SentencesAnnotation.class);
			//String sentiment= " ";
			int count = 0;
			double rating = 0.0;
			for (CoreMap sentence : sentences) {
				String sentiment = sentence.get(SentimentCoreAnnotations.SentimentClass.class);
				if(sentiment.equals("Very negative"))
				{
				  rating += 1.0;
				}
				else if (sentiment.equals("Negative"))
				{
				  rating += 2.0;
				}
				else if (sentiment.equals("Neutral"))
				{
				  rating += 3.0;
				}
				else if (sentiment.equals("Positive"))
				{
				  rating += 4.0;
				}
				else if (sentiment.equals("Very positive"))
				{
				  rating += 5.0;
				}
				count++;
			//	context.write(new Text(String.format("C: '%d'; Ra: '%f'", count, rating)), null);
			}
			rate = new DoubleWritable((double)rating/(double)count);
		 	// emit the tuple and the original contents of the line
			//context.write(new Text(String.format("Business ID: '%s'", bus_id)), null);
			context.write(new Text(String.format(bus_id)), rate);
			//context.write(new Text(String.format("Rating: '%f'",(double)rating/(double)count)), null);
			// emit the tuple and the original contents of the line
			//context.write(new Text(String.format("Rating: '%s'", sentiment)), null);
		 } catch(ParseException e) {
		 	context.write(new Text(String.format("Something went wrong while parsing")), null);
		 }
	}
  }
  
  public static class RatingReducer
       extends Reducer<Text,DoubleWritable,Text,DoubleWritable> {
    private DoubleWritable result = new DoubleWritable();

    public void reduce(Text key, Iterable<DoubleWritable> values,
                       Context context
                       ) throws IOException, InterruptedException {
      double sum = 0.0;
      double count = 0.0;
      for (DoubleWritable val : values) {
        sum = sum + val.get();
        count = count + 1.0;
      }
      result.set(sum/count);
      context.write(key, result);
    }
  }
}
